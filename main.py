import json
from loguru import logger
from config import cursor, client, conn
from services.classify import Classify

classifier = Classify()

def run_kb(user_input: str):
    # Embed query
    print("Generating embedding for query...")
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_input
    ).data[0].embedding

    try:
        conn.rollback()
        logger.info("Transaction rolled back successfully")
    except Exception as e:
        logger.warning(f"Rollback warning: {e}")

    # Retrieve top submissions
    logger.info("Retrieving top matching submissions...")
    try:
        cursor.execute("""
            SELECT submission_id, domain_id, title, selftext, permalink
            FROM submissions
            ORDER BY embedding <-> %s::vector
            LIMIT 3;
        """, (json.dumps(q_emb),))
        top_subs = cursor.fetchall()

        if not top_subs:
            logger.warning("No relevant submissions found.")
        else:
            logger.info(f"Found {len(top_subs)} relevant submissions.")

        # Collect related comments
        context_blocks = []
        sources = []  # track all permalinks for final output
        
        for sid, domain_id, title, selftext, permalink in top_subs:
            cursor.execute("""
                SELECT body, score
                FROM comments
                WHERE submission_id = %s
                ORDER BY embedding <-> %s::vector
                LIMIT 5;
            """, (sid, json.dumps(q_emb)))
            comments = cursor.fetchall()

            joined_comments = "\n".join([f"- {c[0][:600]}" for c in comments])
            block = f"""
    [POST] "{title}"
    URL: {permalink}

    {selftext[:1200] if selftext else "(No text body)"}

    Top Comments:
    {joined_comments}
    """
            context_blocks.append(block)
            sources.append(f"- {title}: https://reddit.com{permalink}")

        context_text = "\n\n---\n\n".join(context_blocks)

        print("\n" + "="*80)
        print("RETRIEVED CONTEXT FROM KB")
        print("="*80)
        print(context_text)
        print("="*80 + "\n")

        # LLM synthesis
        print("Generating output...")
        prompt = f"""
    You are an empathetic assistant that uses examples from real Reddit users' discussions
    to guide someone based on their current life goal.

    User Query:
    {user_input}

    Here are similar cases from Reddit:
    {context_text}

    Using the experiences and insights quoted above, write an actionable plan
    to help the user.
    At the END of your response, include a "Sources" section listing all the Reddit discussions you referenced.
    Format each source as: "- [Post Title]: [full permalink URL]"
    """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful, grounded life-strategy coach."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        print("\n--- LLM RESPONSE ---\n")
        print(response.choices[0].message.content)
        
        # === ensure sources are always visible ===
        print("\n" + "="*80)
        print("DISCUSSION SOURCES")
        print("="*80)
        for src in sources:
            print(src)
        print("="*80 + "\n")
        
        conn.commit()

    except Exception as e:
        logger.error(f"Error during query execution: {e}")
        conn.rollback()
        raise


def main():
    print("Hi")
    # user_input = input("Enter your query or goal: ").strip()
    # run_kb(user_input=user_input)

    # Test Pred
    sample_comments = [
        "This advice actually worked for me, highly recommend!",
        "Don't follow this, it didn't help at all.",
        "I tried it and it was okay, maybe try something else."
    ]

    results = classifier.predict(sample_comments)
    for result in results:
        print(result)
        
if __name__ == '__main__':
    main()