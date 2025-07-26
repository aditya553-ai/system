import os
import json
import redis
import time
from typing import Dict, List, Any, Optional, Tuple
from .hybrid_entity_mapper import HybridEntityMapper

class CachedEntityMapper:
    """
    Performance-optimized entity mapper that uses Redis caching
    to store and retrieve previously mapped entities.
    """
    
    def __init__(self, redis_host: str = None, redis_port: int = None, ttl: int = 2592000):
        """
        Initialize the cached entity mapper.
        
        Args:
            redis_host: Redis host (defaults to REDIS_HOST env var or 'localhost')
            redis_port: Redis port (defaults to REDIS_PORT env var or 6379)
            ttl: Time-to-live for cached entries in seconds (default: 30 days)
        """
        self.hybrid_mapper = HybridEntityMapper(use_cache=False)  # Don't use internal caching
        self.ttl = ttl  # Cache TTL in seconds
        
        # Initialize Redis client
        try:
            host = redis_host or os.getenv('REDIS_HOST', 'localhost')
            port = redis_port or int(os.getenv('REDIS_PORT', 6379))
            self.redis = redis.Redis(host=host, port=port, db=0)
            self.redis.ping()  # Test connection
            print(f"CachedEntityMapper connected to Redis at {host}:{port}")
            self.cache_available = True
        except Exception as e:
            print(f"Redis cache initialization failed: {e}")
            print("CachedEntityMapper will operate without caching")
            self.cache_available = False

    def map_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map entities with caching for performance optimization.
        
        Args:
            entities: List of entities to map
            
        Returns:
            List of mapped entities
        """
        if not entities:
            return []
            
        mapped_entities = []
        uncached_entities = []
        
        # Check cache for each entity
        for entity in entities:
            cached_mapping = self._get_from_cache(entity)
            
            if cached_mapping:
                # Use cached mapping
                entity.update(cached_mapping)
                entity['source'] = f"{entity.get('source', 'unknown')}_cached"
                mapped_entities.append(entity)
            else:
                # Need to map this entity
                uncached_entities.append(entity)
        
        # Process uncached entities in batch if any exist
        if uncached_entities:
            newly_mapped = self.hybrid_mapper.map_entities(uncached_entities)
            
            # Cache the newly mapped entities
            for entity in newly_mapped:
                self._add_to_cache(entity)
                mapped_entities.append(entity)
        
        return mapped_entities
        
    def _get_from_cache(self, entity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get entity mapping from cache.
        
        Args:
            entity: Entity to look up in cache
            
        Returns:
            Cached mapping if available, None otherwise
        """
        if not self.cache_available:
            return None
            
        # Create a cache key from entity text and type
        cache_key = f"entity_map:{entity['label']}:{entity['text']}"
        
        try:
            cached_data = self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            print(f"Error retrieving from cache: {e}")
            
        return None
        
    def _add_to_cache(self, entity: Dict[str, Any]) -> bool:
        """
        Add entity mapping to cache.
        
        Args:
            entity: Mapped entity to cache
            
        Returns:
            True if successfully cached, False otherwise
        """
        if not self.cache_available:
            return False
            
        cache_key = f"entity_map:{entity['label']}:{entity['text']}"
        
        # Extract cacheable data (avoid storing redundant info)
        cacheable_data = {
            'preferred_term': entity.get('preferred_term'),
            'code_systems': entity.get('code_systems'),
            'confidence': entity.get('confidence'),
            'verified': entity.get('verified', False),
            'verification_confidence': entity.get('verification_confidence'),
            'cache_time': time.time()
        }
        
        try:
            self.redis.setex(
                cache_key,
                self.ttl,
                json.dumps(cacheable_data)
            )
            return True
        except Exception as e:
            print(f"Error adding to cache: {e}")
            return False
    
    def clear_cache(self, entity_type: Optional[str] = None) -> int:
        """
        Clear entity mapping cache.
        
        Args:
            entity_type: Optional entity type to clear (clears all if None)
            
        Returns:
            Number of cache entries cleared
        """
        if not self.cache_available:
            return 0
            
        try:
            pattern = f"entity_map:{entity_type}:*" if entity_type else "entity_map:*"
            keys = self.redis.keys(pattern)
            if keys:
                deleted = self.redis.delete(*keys)
                return deleted
            return 0
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return 0