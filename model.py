import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, TransformerEncoder, TransformerEncoderLayer
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import time
from collections import defaultdict
import math

class MultimodalEpisodeEncoder(nn.Module):
    """
    Multimodal Episode Encoder (MEE) that transforms multimodal inputs 
    into compressed episodic representations with cross-modal temporal binding.
    """
    
    def __init__(self, visual_dim=512, audio_dim=256, text_dim=768, 
                 fusion_dim=1024, num_heads=8, num_layers=6):
        super().__init__()
        
        # Individual modality encoders
        self.visual_encoder = nn.Sequential(
            nn.Linear(2048, visual_dim),  # Assuming ResNet-50 features
            nn.ReLU(),
            nn.LayerNorm(visual_dim)
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(768, audio_dim),  # Assuming Wav2Vec2 features
            nn.ReLU(),
            nn.LayerNorm(audio_dim)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(768, text_dim),  # BERT features
            nn.ReLU(),
            nn.LayerNorm(text_dim)
        )
        
        # Cross-modal temporal binding
        total_dim = visual_dim + audio_dim + text_dim
        self.cross_modal_projection = nn.Linear(total_dim, fusion_dim)
        
        # Multi-head attention for cross-modal binding
        self.cross_modal_attention = MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Hierarchical transformer layers
        encoder_layer = TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            batch_first=True
        )
        
        self.transformer1 = TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer2 = TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer3 = TransformerEncoder(encoder_layer, num_layers=2)
        
        # Temporal position encoding parameters
        self.fusion_dim = fusion_dim
        self.temporal_decay = 0.1
        
    def create_temporal_position_encoding(self, seq_len, elapsed_time=0):
        """Create temporal position encoding with decay factors."""
        pe = torch.zeros(seq_len, self.fusion_dim)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.fusion_dim, 2).float() * 
                           -(math.log(10000.0) / self.fusion_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Apply temporal decay
        decay_factor = torch.exp(-self.temporal_decay * elapsed_time)
        pe *= decay_factor
        
        return pe.unsqueeze(0)
    
    def forward(self, visual_input, audio_input, text_input, context=None, elapsed_time=0):
        """
        Forward pass through the multimodal episode encoder.
        
        Args:
            visual_input: Visual features [batch_size, visual_features]
            audio_input: Audio features [batch_size, audio_features]  
            text_input: Text features [batch_size, text_features]
            context: Additional context information
            elapsed_time: Time elapsed since episode creation
        """
        batch_size = visual_input.size(0)
        
        # Encode individual modalities
        v_encoded = self.visual_encoder(visual_input)
        a_encoded = self.audio_encoder(audio_input)
        t_encoded = self.text_encoder(text_input)
        
        # Concatenate modality representations
        multimodal_repr = torch.cat([v_encoded, a_encoded, t_encoded], dim=-1)
        
        # Project to fusion dimension
        fused = self.cross_modal_projection(multimodal_repr)
        fused = fused.unsqueeze(1)  # Add sequence dimension
        
        # Cross-modal temporal binding
        tau, _ = self.cross_modal_attention(fused, fused, fused)
        
        # Add temporal position encoding
        seq_len = tau.size(1)
        pos_encoding = self.create_temporal_position_encoding(seq_len, elapsed_time)
        if tau.device != pos_encoding.device:
            pos_encoding = pos_encoding.to(tau.device)
        tau = tau + pos_encoding
        
        # Hierarchical encoding at multiple temporal scales
        # Level 1: Immediate context
        e1 = self.transformer1(tau)
        
        # Level 2: Medium-term patterns (simulated by repeating)
        e2_input = e1.repeat(1, 5, 1)  # Simulate window of 5
        e2 = self.transformer2(e2_input)
        
        # Level 3: Long-term patterns (simulated by repeating)
        e3_input = e2.repeat(1, 5, 1)  # Simulate window of 25
        e3 = self.transformer3(e3_input)
        
        # Return final episode representation
        episode_repr = e3.mean(dim=1)  # Global average pooling
        
        return {
            'episode_embedding': episode_repr,
            'level1': e1.squeeze(1),
            'level2': e2.mean(dim=1),
            'level3': e3.mean(dim=1),
            'temporal_tokens': tau.squeeze(1)
        }


class MemoryBankStorage(nn.Module):
    """
    Three-tier hierarchical storage system: Hot, Warm, and Cold memory.
    """
    
    def __init__(self, hot_capacity=1000, warm_capacity=5000, cold_capacity=50000,
                 episode_dim=1024, compressed_dim=128):
        super().__init__()
        
        self.hot_capacity = hot_capacity
        self.warm_capacity = warm_capacity  
        self.cold_capacity = cold_capacity
        self.episode_dim = episode_dim
        self.compressed_dim = compressed_dim
        
        # Hot memory - full resolution storage
        self.hot_memory = {}
        self.hot_timestamps = {}
        self.hot_access_counts = defaultdict(int)
        
        # Warm memory - compressed storage
        self.warm_memory = {}
        self.warm_timestamps = {}
        self.warm_access_counts = defaultdict(int)
        
        # Cold memory - semantic summaries
        self.cold_memory = {}
        self.cold_timestamps = {}
        self.cold_access_counts = defaultdict(int)
        
        # Compression networks
        self.visual_compressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.audio_compressor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(), 
            nn.Linear(64, 32)
        )
        
        self.text_compressor = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.cross_modal_fusion = nn.Sequential(
            nn.Linear(64 + 32 + 64 + episode_dim, compressed_dim),
            nn.ReLU(),
            nn.LayerNorm(compressed_dim)
        )
        
        # Consolidation parameters
        self.alpha = 0.3  # Access frequency weight
        self.beta = 0.4   # Cross-modal consistency weight
        self.gamma = 0.3  # Semantic relevance weight
        
    def compute_retention_score(self, episode_id, tier='hot'):
        """Compute retention score for consolidation decisions."""
        if tier == 'hot':
            access_count = self.hot_access_counts[episode_id]
            timestamp = self.hot_timestamps.get(episode_id, time.time())
        elif tier == 'warm':
            access_count = self.warm_access_counts[episode_id]
            timestamp = self.warm_timestamps.get(episode_id, time.time())
        else:
            access_count = self.cold_access_counts[episode_id]
            timestamp = self.cold_timestamps.get(episode_id, time.time())
            
        # Access frequency component
        f_access = math.log(1 + access_count)
        
        # Cross-modal consistency (simplified)
        s_consistency = 0.7  # Placeholder
        
        # Semantic relevance (simplified)
        r_relevance = 0.6  # Placeholder
        
        retention_score = (self.alpha * f_access + 
                         self.beta * s_consistency + 
                         self.gamma * r_relevance)
        
        return retention_score
    
    def compress_episode(self, episode_data):
        """Compress episode for warm memory storage."""
        # Extract modality components (assuming they're available)
        visual_features = episode_data.get('visual', torch.zeros(512))
        audio_features = episode_data.get('audio', torch.zeros(256))
        text_features = episode_data.get('text', torch.zeros(768))
        episode_embedding = episode_data['episode_embedding']
        
        # Compress individual modalities
        z_v = self.visual_compressor(visual_features)
        z_a = self.audio_compressor(audio_features)
        z_t = self.text_compressor(text_features)
        
        # Fuse compressed representations
        combined = torch.cat([z_v, z_a, z_t, episode_embedding], dim=-1)
        compressed = self.cross_modal_fusion(combined)
        
        return compressed
        
    def store_episode(self, episode_id, episode_data, tier='hot'):
        """Store episode in specified memory tier."""
        current_time = time.time()
        
        if tier == 'hot':
            if len(self.hot_memory) >= self.hot_capacity:
                self._evict_from_hot()
            self.hot_memory[episode_id] = episode_data
            self.hot_timestamps[episode_id] = current_time
            
        elif tier == 'warm':
            if len(self.warm_memory) >= self.warm_capacity:
                self._evict_from_warm()
            compressed_episode = self.compress_episode(episode_data)
            self.warm_memory[episode_id] = compressed_episode
            self.warm_timestamps[episode_id] = current_time
            
        elif tier == 'cold':
            if len(self.cold_memory) >= self.cold_capacity:
                self._evict_from_cold()
            # Create semantic summary (simplified)
            summary = self._create_semantic_summary(episode_data)
            self.cold_memory[episode_id] = summary
            self.cold_timestamps[episode_id] = current_time
    
    def _evict_from_hot(self):
        """Evict least valuable episode from hot memory."""
        if not self.hot_memory:
            return
            
        # Find episode with lowest retention score
        min_score = float('inf')
        min_episode_id = None
        
        for episode_id in self.hot_memory:
            score = self.compute_retention_score(episode_id, 'hot')
            if score < min_score:
                min_score = score
                min_episode_id = episode_id
        
        if min_episode_id:
            # Move to warm memory
            episode_data = self.hot_memory.pop(min_episode_id)
            self.hot_timestamps.pop(min_episode_id)
            self.store_episode(min_episode_id, episode_data, 'warm')
    
    def _evict_from_warm(self):
        """Evict least valuable episode from warm memory."""
        if not self.warm_memory:
            return
            
        min_score = float('inf')
        min_episode_id = None
        
        for episode_id in self.warm_memory:
            score = self.compute_retention_score(episode_id, 'warm')
            if score < min_score:
                min_score = score
                min_episode_id = episode_id
        
        if min_episode_id:
            # Move to cold memory (create summary)
            episode_data = self.warm_memory.pop(min_episode_id)
            self.warm_timestamps.pop(min_episode_id)
            summary_data = {'episode_embedding': episode_data}  # Simplified
            self.store_episode(min_episode_id, summary_data, 'cold')
    
    def _evict_from_cold(self):
        """Evict least valuable episode from cold memory."""
        if not self.cold_memory:
            return
            
        min_score = float('inf')
        min_episode_id = None
        
        for episode_id in self.cold_memory:
            score = self.compute_retention_score(episode_id, 'cold')
            if score < min_score:
                min_score = score
                min_episode_id = episode_id
        
        if min_episode_id:
            # Permanently delete
            self.cold_memory.pop(min_episode_id)
            self.cold_timestamps.pop(min_episode_id)
    
    def _create_semantic_summary(self, episode_data):
        """Create semantic summary for cold storage."""
        # Simplified summary creation
        return {
            'episode_embedding': episode_data['episode_embedding'],
            'summary_timestamp': time.time(),
            'summary_type': 'semantic'
        }
    
    def retrieve_episode(self, episode_id):
        """Retrieve episode from memory, checking all tiers."""
        # Check hot memory first
        if episode_id in self.hot_memory:
            self.hot_access_counts[episode_id] += 1
            return self.hot_memory[episode_id], 'hot'
        
        # Check warm memory
        if episode_id in self.warm_memory:
            self.warm_access_counts[episode_id] += 1
            return self.warm_memory[episode_id], 'warm'
        
        # Check cold memory
        if episode_id in self.cold_memory:
            self.cold_access_counts[episode_id] += 1
            return self.cold_memory[episode_id], 'cold'
        
        return None, None


class ExperienceGuidedRetrieval(nn.Module):
    """
    Context-aware retrieval of relevant past experiences for current decision-making.
    """
    
    def __init__(self, embed_dim=1024, similarity_threshold=0.7):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.similarity_threshold = similarity_threshold
        
        # Context embedding function
        self.context_embedding = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Modality confidence estimators
        self.visual_confidence = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.audio_confidence = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.text_confidence = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(), 
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Temporal weighting parameters
        self.temporal_decay = 0.1
        self.relevance_boost = 0.2
        
    def compute_similarity(self, episode_embedding, current_context):
        """Compute cosine similarity between episode and current context."""
        # Embed current context
        context_embedded = self.context_embedding(current_context)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(
            episode_embedding.unsqueeze(0) if episode_embedding.dim() == 1 else episode_embedding,
            context_embedded.unsqueeze(0) if context_embedded.dim() == 1 else context_embedded,
            dim=-1
        )
        
        return similarity
    
    def compute_partial_similarity(self, episode_data, query_data, available_modalities):
        """Compute similarity with partial modality queries."""
        total_similarity = 0.0
        total_weight = 0.0
        
        modality_weights = {}
        
        # Compute confidence weights for available modalities
        if 'visual' in available_modalities and 'visual' in query_data:
            confidence = self.visual_confidence(query_data['visual'])
            modality_weights['visual'] = F.softmax(confidence, dim=0)
        
        if 'audio' in available_modalities and 'audio' in query_data:
            confidence = self.audio_confidence(query_data['audio'])
            modality_weights['audio'] = F.softmax(confidence, dim=0)
            
        if 'text' in available_modalities and 'text' in query_data:
            confidence = self.text_confidence(query_data['text'])
            modality_weights['text'] = F.softmax(confidence, dim=0)
        
        # Compute weighted similarity
        for modality in available_modalities:
            if modality in episode_data and modality in query_data:
                sim = F.cosine_similarity(
                    episode_data[modality].unsqueeze(0) if episode_data[modality].dim() == 1 else episode_data[modality],
                    query_data[modality].unsqueeze(0) if query_data[modality].dim() == 1 else query_data[modality],
                    dim=-1
                )
                weight = modality_weights.get(modality, torch.tensor(1.0))
                total_similarity += weight * sim
                total_weight += weight
        
        if total_weight > 0:
            return total_similarity / total_weight
        else:
            return torch.tensor(0.0)
    
    def compute_temporal_weight(self, timestamp, relevance_score=0.5):
        """Compute temporal weighting with decay and relevance boost."""
        current_time = time.time()
        elapsed_time = current_time - timestamp
        
        temporal_weight = torch.exp(torch.tensor(-self.temporal_decay * elapsed_time))
        relevance_boost = self.relevance_boost * relevance_score
        
        return temporal_weight + relevance_boost
    
    def hierarchical_search(self, memory_bank, current_context, k_min=5, k_max=20):
        """Perform hierarchical search across memory tiers."""
        candidates = []
        
        # Search hot memory first
        hot_candidates = self._search_memory_tier(
            memory_bank.hot_memory, 
            memory_bank.hot_timestamps,
            current_context,
            'hot'
        )
        candidates.extend(hot_candidates)
        
        # Search warm memory if needed
        if len(candidates) < k_min:
            warm_candidates = self._search_memory_tier(
                memory_bank.warm_memory,
                memory_bank.warm_timestamps, 
                current_context,
                'warm'
            )
            candidates.extend(warm_candidates)
        
        # Search cold memory if still needed
        if len(candidates) < k_min:
            cold_candidates = self._search_memory_tier(
                memory_bank.cold_memory,
                memory_bank.cold_timestamps,
                current_context, 
                'cold'
            )
            candidates.extend(cold_candidates)
        
        # Rank by relevance and return top-k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:k_max]
    
    def _search_memory_tier(self, memory_dict, timestamp_dict, current_context, tier):
        """Search specific memory tier for relevant episodes."""
        candidates = []
        
        for episode_id, episode_data in memory_dict.items():
            # Extract episode embedding
            if isinstance(episode_data, dict):
                episode_embedding = episode_data.get('episode_embedding', episode_data)
            else:
                episode_embedding = episode_data
            
            # Compute similarity
            similarity = self.compute_similarity(episode_embedding, current_context)
            
            # Apply temporal weighting  
            timestamp = timestamp_dict.get(episode_id, time.time())
            temporal_weight = self.compute_temporal_weight(timestamp)
            
            # Compute final score
            final_score = similarity * temporal_weight
            
            if final_score > self.similarity_threshold:
                candidates.append({
                    'episode_id': episode_id,
                    'episode_data': episode_data,
                    'similarity': similarity,
                    'temporal_weight': temporal_weight,
                    'score': final_score,
                    'tier': tier
                })
        
        return candidates


class TMMB(nn.Module):
    """
    Main Temporal Multimodal Memory Banks architecture.
    """
    
    def __init__(self, visual_dim=512, audio_dim=256, text_dim=768, 
                 fusion_dim=1024, num_heads=8, num_layers=6,
                 hot_capacity=1000, warm_capacity=5000, cold_capacity=50000):
        super().__init__()
        
        # Core components
        self.episode_encoder = MultimodalEpisodeEncoder(
            visual_dim=visual_dim,
            audio_dim=audio_dim, 
            text_dim=text_dim,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        self.memory_bank = MemoryBankStorage(
            hot_capacity=hot_capacity,
            warm_capacity=warm_capacity,
            cold_capacity=cold_capacity,
            episode_dim=fusion_dim
        )
        
        self.retrieval_system = ExperienceGuidedRetrieval(
            embed_dim=fusion_dim
        )
        
        self.episode_counter = 0
        
    def store_experience(self, visual_input, audio_input, text_input, context=None):
        """Store a new multimodal experience."""
        # Encode episode
        episode_data = self.episode_encoder(
            visual_input, audio_input, text_input, context
        )
        
        # Generate episode ID
        episode_id = f"episode_{self.episode_counter}"
        self.episode_counter += 1
        
        # Store in hot memory
        self.memory_bank.store_episode(episode_id, episode_data, tier='hot')
        
        return episode_id, episode_data
    
    def retrieve_relevant_experiences(self, current_context, k=10):
        """Retrieve relevant past experiences for current context."""
        relevant_episodes = self.retrieval_system.hierarchical_search(
            self.memory_bank, 
            current_context,
            k_max=k
        )
        
        return relevant_episodes
    
    def forward(self, visual_input, audio_input, text_input, context=None, 
                retrieve_memories=True, k_retrieve=10):
        """
        Forward pass: encode current experience and optionally retrieve relevant memories.
        """
        # Encode current episode
        current_episode = self.episode_encoder(
            visual_input, audio_input, text_input, context
        )
        
        retrieved_memories = []
        if retrieve_memories:
            # Use current episode embedding as query context
            query_context = current_episode['episode_embedding']
            retrieved_memories = self.retrieve_relevant_experiences(
                query_context, k=k_retrieve
            )
        
        return {
            'current_episode': current_episode,
            'retrieved_memories': retrieved_memories,
            'num_retrieved': len(retrieved_memories)
        }
    
    def consolidate_memories(self):
        """Trigger memory consolidation process."""
        # This would typically run as a background process
        # For now, just trigger evictions if at capacity
        if len(self.memory_bank.hot_memory) >= self.memory_bank.hot_capacity:
            self.memory_bank._evict_from_hot()
        
        if len(self.memory_bank.warm_memory) >= self.memory_bank.warm_capacity:
            self.memory_bank._evict_from_warm()
            
        if len(self.memory_bank.cold_memory) >= self.memory_bank.cold_capacity:
            self.memory_bank._evict_from_cold()
    
    def get_memory_stats(self):
        """Get current memory usage statistics."""
        return {
            'hot_memory_size': len(self.memory_bank.hot_memory),
            'warm_memory_size': len(self.memory_bank.warm_memory),
            'cold_memory_size': len(self.memory_bank.cold_memory),
            'total_episodes': self.episode_counter
        }
