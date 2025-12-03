import json
from typing import List, Dict, Any
from openai import OpenAI

# Reset on reddit-pipeline
class QueryPipeline:
    def __init__(self, llm_client: OpenAI, llm_model_name: str, cursor):
        # LLM instantiations
        self.llm_model_name = llm_model_name
        self.llm_client = llm_client

        # Load domains.json
        with open("domains.json", "r") as f:
            self.domains = json.load(f)

        # PG DB Conn instantiations
        self.dbcursor = cursor
    
    def __generate_HyDE_emb(self, user_story: str, user_current_steps: str) -> Dict[str, Any]:
        """
        The idea is to provide next steps or recommended guidance on next steps
        Given the current steps taken by the user & the context of the user story
        """

        # Prepare JSON for classifier
        domains_json_string = json.dumps(self.domains, indent=2)

        HyDE_prompt = f"""
        You are generating a *hypothetical Reddit post* for retrieval augmentation (HyDE).
        Your goal is to restate the user’s situation in an emotionally clear, concrete,
        and detailed way so the embedding captures all relevant dimensions.

        Requirements:
        - include a short realistic title
        - 150–200 word body
        - describe the user's background and current steps factually
        - describe emotions, obstacles, fears, and uncertainty
        - every sentence should add unique, concrete information
        - use specific nouns and verbs, avoid vague filler
        - do NOT provide advice, solutions, or resolutions

        User background/context:
        {user_story}

        Current steps they've already taken:
        {user_current_steps}

        ----

        Classify the user’s situation into one or more relevant domains.
        Here are the available domains:

        {domains_json_string}

        You MUST respond in EXACTLY the following JSON format.
        Do not add comments, explanations, or extra text.

        {{
        "hyde_post": "<150-200 word post>",
        "domains": ["<list of matching domain tags from the domain_tag in domains object>"]
        }}
        """


            # Run LLM
        response = self.llm_client.chat.completions.create(
            model=self.llm_model_name,
            messages=[
                {"role": "system", "content": "You are a Reddit user asking for help in a detailed, vulnerable way."},
                {"role": "user", "content": HyDE_prompt}
            ],
            temperature=0.7,
            max_tokens=700
        )

        # Parse JSON
        parsed = json.loads(response.choices[0].message.content)
        domains = parsed["domains"]
        hyde_post = parsed["hyde_post"]

        # Generate embedding
        HyDE_emb = self.llm_client.embeddings.create(
            model="text-embedding-3-small",
            input=hyde_post
        ).data[0].embedding

        return {
            "HyDE_emb": HyDE_emb, 
            "Domains": domains
        }
    
    def __run_submission_document_retrieval(self, HyDE_emb: List[float], domains: List[str]) -> List[tuple[Any, ...]]:
        print("Retrieveing top matching submissions....")
        self.cursor.execute("""
                SELECT 
                    s.submission_id,
                    s.domain_id,
                    s.title,
                    s.selftext,
                    s.permalink,
                    (s.embedding <=> %s::vector) AS distance,
                FROM submissions s
                JOIN domain d ON s.domain_id = d.domain_id
                WHERE d.domain_tag = ANY(%s)
                ORDER BY s.embedding <=> %s::vector
                LIMIT 15;
            """, (json.dumps(HyDE_emb), domains, json.dumps(HyDE_emb)))
        top_subs = self.cursor.fetchall()
        # Filter by threshold requirement
        top_subs_filtered = [subs for subs in top_subs if subs[5] < 0.45]
        
        if not top_subs:
            print("❌ No Relevant submissions found")
        else:
            print(f"✅ Found {len(top_subs)} relevant submissions")

        # Printing submissions
        return top_subs_filtered
    
    def __run_comment_document_retreival(self, HyDE_emb: List[float], top_subs_filtered: List[tuple[Any, ...]]) -> dict:
        top_submission_ids = [row[0] for row in top_subs_filtered]
        print(f"Current Top Submission ids --> {json.dumps(top_submission_ids, indent=2)}")

        # Pulling in the most Relevant comments from only the top posts that we get
        # There is also a small scoring boost for the high upvoted comments, at the time of querying
        # Relevance first tho, only then popularity
        query = """
        WITH ranked_comments AS (
            SELECT 
                c.body,
                c.score,
                s.title,
                c.submission_id,
                s.permalink,
                c.embedding <=> %s::vector AS distance
            FROM comments c
            JOIN submissions s ON c.submission_id = s.submission_id
            WHERE c.submission_id = ANY(%s)
        )
        SELECT 
            body,
            score,
            title,
            submission_id,
            permalink,
            distance
        FROM ranked_comments
        ORDER BY 
            distance + (1.0 / (score + 10)) ASC 
        LIMIT 20;
        """

        # Note the order of parameters is now --> embedding, submission_ids list
        self.cursor.execute(query, (json.dumps(HyDE_emb), top_submission_ids))
        relevant_comments = self.cursor.fetchall()
        print(f"\n✅ Retrieved {len(relevant_comments)} relevant comments\n")

        # Context map contains : submission_id -> {submission_info, comments}
        context_map = {}
        # Append submission details to context map
        for sub in top_subs_filtered:
            sub_id = sub[0]
            sub_title = sub[2]
            sub_selftext = sub[3]
            sub_permalink = sub[4]

            if sub_id not in context_map:
                context_map[sub_id] = {
                    'title': sub_title,
                    'permalink': sub_permalink,
                    'selftext': sub_selftext,
                    'comments': []
                }
        
        # Append comments to context map
        for comment in relevant_comments:
            body = comment[0]
            score = comment[1]
            submission_id = comment[3]
            distance = comment[5]
            context_map[submission_id]['comments'].append({
                'body': body,
                'score': score,
                'distance': distance
            })
        
        # Ready for LLM chain call
        return context_map
    
    def generate_actionable_steps(self, user_story: str, user_current_steps: str):
        """
        Full actionable steps pipeline
        """

        # Genenate hyde_post + embeddings, with domains
        HyDE_emb, domains = self.__generate_HyDE_emb(
        user_story=user_story,
        user_current_steps=user_current_steps
        )

        # Retreive top relevant submissions
        top_subs = self.__run_submission_document_retrieval(
            HyDE_emb=HyDE_emb,
            domains=domains
        )

        if not top_subs:
            return {
                "status": "no_matches",
                "message": "No relevant submissions found under the confidence threshold, ***RUN WEBAPI PIPELINE***"
            }

        # Retreive comments to relevant top subs to create context map
        context_map = self.__run_comment_document_retreival(
            HyDE_emb=HyDE_emb,
            top_subs_filtered=top_subs,
        )

        all_steps = []  # will hold dict: {step, emb, submission_id, url, score}
        # Actionable step extraction per comment for LLM chain
        for sub_id, sub_data in context_map.items():
            print(f"Deriving Actionable steps for Submission -> {sub_data['title']}[{sub_data[sub_id]}]")

            action_step_prompt = """
                You will be given a single Reddit comment.
                Extract ONLY clear actionable steps expressed or strongly implied.
                Do NOT add new ideas. Do NOT give advice. Do NOT restate context.
                Return JSON exactly as: {"steps": ["...", "..."]}

                COMMENT:
            """
            # Generating an action step for each comment
            for c in sub_data['comments']:
                comment_body = c["body"]
                score = c["score"]

                # LLM: Extract steps
                resp = self.llm_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[
                        {"role": "system", "content": "Extract only actionable, literal steps."},
                        {"role": "user", "content": action_step_prompt + comment_body}
                    ],
                    temperature=0.2, # Follow a lil bit more strict
                    max_tokens=120 # Nothing too crazy of a step
                )

                try:
                    parsed = json.loads(resp.choices[0].message.content)
                    steps = parsed.get("steps", [])
                except:
                    steps = []
                
                # For each extracted step → embed immediately
                for step_text in steps:
                    emb = self.llm_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=step_text
                    ).data[0].embedding

                    all_steps.append({
                        "step": step_text,
                        "embedding": emb,
                        "submission_id": sub_id,
                        "permalink": sub_data['permalink'],
                        "score": score
                    })
            
        if not all_steps:
            return {
                "status": "empty",
                "message": "Steps extraction failed — maybe comments didn’t contain actionable steps."
            }



