import os
import asyncpg
import asyncio
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()


class Database:
    """Async PostgreSQL database manager optimized for local Docker DB."""
    
    def __init__(
        self,
        dsn: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_size: int = 5,
        max_size: int = 20,
        command_timeout: int = 60,
        statement_cache_size: int = 100
    ):
        """
        Initialize database connection manager.
        
        Args:
            dsn: Full PostgreSQL connection string. If provided, other connection params ignored.
            host: Database host (default: localhost or POSTGRES_HOST env var)
            port: Database port (default: 5432 or POSTGRES_PORT env var)
            database: Database name (default: postgres or POSTGRES_DB env var)
            user: Database user (default: postgres or POSTGRES_USER env var)
            password: Database password (default: postgres or POSTGRES_PASSWORD env var)
            min_size: Minimum connections in pool
            max_size: Maximum connections in pool
            command_timeout: Query timeout in seconds
            statement_cache_size: Statement cache size
        """
        # Build DSN from components if not provided
        if dsn:
            self.dsn = dsn
        else:
            host = host or os.environ.get("POSTGRES_HOST", "localhost")
            port = port or int(os.environ.get("POSTGRES_PORT", "5432"))
            database = database or os.environ.get("POSTGRES_DB", "postgres")
            user = user or os.environ.get("POSTGRES_USER", "postgres")
            password = password or os.environ.get("POSTGRES_PASSWORD", "postgres")
            
            self.dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        self.min_size = min_size
        self.max_size = max_size
        self.command_timeout = command_timeout
        self.statement_cache_size = statement_cache_size
        self.pool: Optional[asyncpg.Pool] = None
        self._pool_lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the connection pool with optimized settings for Docker PostgreSQL."""
        async with self._pool_lock:
            if self.pool is None:
                self.pool = await asyncpg.create_pool(
                    dsn=self.dsn,
                    min_size=self.min_size,
                    max_size=self.max_size,
                    command_timeout=self.command_timeout,
                    statement_cache_size=self.statement_cache_size,
                )
                print(f"DB connection pool initialized (min={self.min_size}, max={self.max_size})")
                
                # Pre-warm the pool
                async with self.pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                print("Database connection pre-warmed")
    
    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            print("Database connection pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Context manager for acquiring a connection from pool."""
        if not self.pool:
            await self.initialize()
        
        conn = await self.pool.acquire()
        try:
            yield conn
        finally:
            await self.pool.release(conn)
    
    async def fetch(
        self,
        query: str,
        *params,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return all rows as a list of dicts.
        
        Args:
            query: SQL query with $1, $2, etc. placeholders
            params: Query parameters
            max_retries: Number of retry attempts on connection errors
        """
        if not self.pool:
            await self.initialize()
        
        for attempt in range(max_retries):
            try:
                async with self.acquire() as conn:
                    rows = await conn.fetch(query, *params)
                    return [dict(row) for row in rows]
            except (asyncpg.PostgresConnectionError, asyncpg.CannotConnectNowError) as e:
                print(f"DB connection error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))
            except Exception as e:
                print(f"DB fetch error: {e}")
                raise
    
    async def fetchrow(
        self,
        query: str,
        *params,
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a SELECT query and return a single row as a dict.
        
        Args:
            query: SQL query with $1, $2, etc. placeholders
            params: Query parameters
            max_retries: Number of retry attempts on connection errors
        """
        if not self.pool:
            await self.initialize()
        
        for attempt in range(max_retries):
            try:
                async with self.acquire() as conn:
                    row = await conn.fetchrow(query, *params)
                    return dict(row) if row else None
            except (asyncpg.PostgresConnectionError, asyncpg.CannotConnectNowError) as e:
                print(f"DB connection error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))
            except Exception as e:
                print(f"DB fetchrow error: {e}")
                raise
    
    async def fetchval(
        self,
        query: str,
        *params,
        max_retries: int = 3
    ) -> Any:
        """
        Execute a query and return a single value.
        
        Args:
            query: SQL query with $1, $2, etc. placeholders
            params: Query parameters
            max_retries: Number of retry attempts on connection errors
        """
        if not self.pool:
            await self.initialize()
        
        for attempt in range(max_retries):
            try:
                async with self.acquire() as conn:
                    return await conn.fetchval(query, *params)
            except (asyncpg.PostgresConnectionError, asyncpg.CannotConnectNowError) as e:
                print(f"DB connection error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))
            except Exception as e:
                print(f"DB fetchval error: {e}")
                raise
    
    async def execute(
        self,
        query: str,
        *params,
        max_retries: int = 3
    ) -> int:
        """
        Execute an INSERT/UPDATE/DELETE statement.
        
        Args:
            query: SQL query with $1, $2, etc. placeholders
            params: Query parameters
            max_retries: Number of retry attempts on connection errors
            
        Returns:
            Number of affected rows
        """
        if not self.pool:
            await self.initialize()
        
        for attempt in range(max_retries):
            try:
                async with self.acquire() as conn:
                    async with conn.transaction():
                        result = await conn.execute(query, *params)
                        # Parse "INSERT 0 1", "UPDATE 3", "DELETE 5"
                        parts = result.split()
                        return int(parts[-1]) if parts[-1].isdigit() else 0
            except (asyncpg.PostgresConnectionError, asyncpg.CannotConnectNowError) as e:
                print(f"DB connection error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))
            except Exception as e:
                print(f"DB execute error: {e}")
                raise
    
    async def executemany(
        self,
        query: str,
        params_list: List[tuple],
        max_retries: int = 3
    ) -> None:
        """
        Execute a query with multiple parameter sets (bulk insert/update).
        
        Args:
            query: SQL query with $1, $2, etc. placeholders
            params_list: List of parameter tuples
            max_retries: Number of retry attempts on connection errors
        """
        if not self.pool:
            await self.initialize()
        
        for attempt in range(max_retries):
            try:
                async with self.acquire() as conn:
                    async with conn.transaction():
                        await conn.executemany(query, params_list)
                        return
            except (asyncpg.PostgresConnectionError, asyncpg.CannotConnectNowError) as e:
                print(f"DB connection error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))
            except Exception as e:
                print(f"DB executemany error: {e}")
                raise
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for transactions."""
        if not self.pool:
            await self.initialize()
        
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn