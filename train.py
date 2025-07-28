import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import json
import logging
from typing import Dict, List, Tuple
import os
from main_tmmb import TMMB

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalDataset(Dataset):
    """Dataset for multimodal episodes."""
    
    def __init__(self, data_path: str):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to the dataset directory
        """
        self.data_path = data_path
        self.episodes = []
        self.load_data()
    
    def load_data(self):
        """Load multimodal episodes from disk."""
        # Placeholder for actual data loading
        # In practice, this would load visual, audio, and text data
        for i in range(1000):  # Dummy data for demonstration
            episode = {
                'visual': torch.randn(2048),  # ResNet-50 features
                'audio': torch.randn(768),    # Wav2Vec2 features
                'text': torch.randn(768),     # BERT features
                'task_label': torch.randint(0, 3, (1,)).item(),  # Task classification
                'episode_id': f'episode_{i}'
            }
            self.episodes.append(episode)
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        return self.episodes[idx]

class TMBBTrainer:
    """Trainer class for TMMB model."""
    
    def __init__(self, model: TMMB, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.alignment_loss = nn.CosineEmbeddingLoss()
        self.task_loss = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('max_epochs', 100)
        )
        
        # Loss weights
        self.alpha = config.get('alignment_weight', 0.1)
        self.beta = config.get('task_weight', 1.0)
        
    def compute_reconstruction_loss(self, encoded, target):
        """Compute reconstruction loss for episode encoding."""
        return self.reconstruction_loss(encoded, target)
    
    def compute_alignment_loss(self, visual_embed, audio_embed, text_embed):
        """Compute cross-modal alignment loss."""
        batch_size = visual_embed.size(0)
        labels = torch.ones(batch_size).to(self.device)
        
        # Visual-Audio alignment
        va_loss = self.alignment_loss(visual_embed, audio_embed, labels)
        
        # Visual-Text alignment  
        vt_loss = self.alignment_loss(visual_embed, text_embed, labels)
        
        # Audio-Text alignment
        at_loss = self.alignment_loss(audio_embed, text_embed, labels)
        
        return (va_loss + vt_loss + at_loss) / 3.0
    
    def compute_task_loss(self, task_predictions, task_labels):
        """Compute task-specific loss."""
        return self.task_loss(task_predictions, task_labels)
    
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            visual_input = batch['visual'].to(self.device)
            audio_input = batch['audio'].to(self.device)
            text_input = batch['text'].to(self.device) 
            task_labels = batch['task_label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                visual_input, audio_input, text_input,
                retrieve_memories=True, k_retrieve=5
            )
            
            current_episode = outputs['current_episode']
            episode_embedding = current_episode['episode_embedding']
            
            # Compute losses
            # 1. Reconstruction loss (simplified - reconstruct from embedding)
            reconstruction_target = torch.cat([
                visual_input, audio_input[:, :512], text_input[:, :512]  # Truncate to match dimensions
            ], dim=-1)
            
            # Project episode embedding to reconstruction space
            reconstruction_proj = nn.Linear(episode_embedding.size(-1), reconstruction_target.size(-1)).to(self.device)
            reconstructed = reconstruction_proj(episode_embedding)
            recon_loss = self.compute_reconstruction_loss(reconstructed, reconstruction_target)
            
            # 2. Cross-modal alignment loss
            # Extract modality-specific embeddings from episode encoding
            visual_embed = current_episode['level1'][:, :512] if current_episode['level1'].size(-1) >= 512 else current_episode['level1']
            audio_embed = current_episode['level1'][:, 512:768] if current_episode['level1'].size(-1) >= 768 else current_episode['level1'][:, :256]
            text_embed = current_episode['level1'][:, -256:] if current_episode['level1'].size(-1) >= 256 else current_episode['level1']
            
            align_loss = self.compute_alignment_loss(visual_embed, audio_embed, text_embed)
            
            # 3. Task-specific loss (simple classification)
            task_classifier = nn.Linear(episode_embedding.size(-1), 3).to(self.device)
            task_predictions = task_classifier(episode_embedding)
            task_loss = self.compute_task_loss(task_predictions, task_labels)
            
            # Combined loss
            total_batch_loss = recon_loss + self.alpha * align_loss + self.beta * task_loss
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update statistics
            total_loss += total_batch_loss.item()
            total_samples += visual_input.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'Align': f'{align_loss.item():.4f}',
                'Task': f'{task_loss.item():.4f}'
            })
            
            # Store experiences in memory
            for i in range(visual_input.size(0)):
                self.model.store_experience(
                    visual_input[i:i+1],
                    audio_input[i:i+1], 
                    text_input[i:i+1]
                )
            
            # Periodic memory consolidation
            if batch_idx % 100 == 0:
                self.model.consolidate_memories()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def evaluate(self, dataloader: DataLoader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        retrieval_accuracies = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                # Move batch to device
                visual_input = batch['visual'].to(self.device)
                audio_input = batch['audio'].to(self.device)
                text_input = batch['text'].to(self.device)
                task_labels = batch['task_label'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    visual_input, audio_input, text_input,
                    retrieve_memories=True, k_retrieve=10
                )
                
                current_episode = outputs['current_episode']
                episode_embedding = current_episode['episode_embedding']
                retrieved_memories = outputs['retrieved_memories']
                
                # Compute task accuracy
                task_classifier = nn.Linear(episode_embedding.size(-1), 3).to(self.device)
                task_predictions = task_classifier(episode_embedding)
                predicted_labels = torch.argmax(task_predictions, dim=1)
                correct_predictions += (predicted_labels == task_labels).sum().item()
                total_samples += visual_input.size(0)
                
                # Compute retrieval quality (simplified)
                if len(retrieved_memories) > 0:
                    # Calculate average similarity of retrieved memories
                    similarities = [mem['similarity'].item() for mem in retrieved_memories]
                    avg_similarity = sum(similarities) / len(similarities)
                    retrieval_accuracies.append(avg_similarity)
        
        accuracy = correct_predictions / total_samples
        avg_retrieval_accuracy = sum(retrieval_accuracies) / len(retrieval_accuracies) if retrieval_accuracies else 0.0
        
        return {
            'accuracy': accuracy,
            'retrieval_accuracy': avg_retrieval_accuracy,
            'memory_stats': self.model.get_memory_stats()
        }
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, 
              num_epochs: int):
        """Full training loop."""
        best_val_accuracy = 0.0
        
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_dataloader, epoch + 1)
            
            # Validation
            val_metrics = self.evaluate(val_dataloader)
            
            # Scheduler step
            self.scheduler.step()
            
            # Logging
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")
            logger.info(f"Epoch {epoch + 1} - Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Epoch {epoch + 1} - Retrieval Accuracy: {val_metrics['retrieval_accuracy']:.4f}")
            logger.info(f"Memory Stats: {val_metrics['memory_stats']}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                self.save_model(f'best_model_epoch_{epoch + 1}.pth')
                logger.info(f"New best model saved with accuracy: {best_val_accuracy:.4f}")
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class BenchmarkEvaluator:
    """Evaluator for benchmark tasks (MultiNav, PersonalAssist, LabAssist)."""
    
    def __init__(self, model: TMMB):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def evaluate_multinav(self, nav_episodes: List[Dict]) -> Dict:
        """Evaluate on MultiNav benchmark task."""
        successful_navigations = 0
        total_navigations = len(nav_episodes)
        
        for episode in nav_episodes:
            # Extract multimodal inputs
            visual_input = torch.tensor(episode['visual']).unsqueeze(0).to(self.device)
            audio_input = torch.tensor(episode['audio']).unsqueeze(0).to(self.device)
            text_input = torch.tensor(episode['text']).unsqueeze(0).to(self.device)
            
            # Model prediction
            with torch.no_grad():
                outputs = self.model(visual_input, audio_input, text_input)
                
            # Simulate navigation success based on retrieved memories
            retrieved_memories = outputs['retrieved_memories']
            if len(retrieved_memories) > 0:
                # Use memory similarity as proxy for navigation success
                avg_similarity = sum([mem['similarity'].item() for mem in retrieved_memories]) / len(retrieved_memories)
                if avg_similarity > 0.7:  # Threshold for successful navigation
                    successful_navigations += 1
        
        success_rate = successful_navigations / total_navigations
        return {'success_rate': success_rate, 'total_episodes': total_navigations}
    
    def evaluate_personal_assist(self, assist_episodes: List[Dict]) -> Dict:
        """Evaluate on PersonalAssist benchmark task."""
        satisfaction_scores = []
        
        for episode in assist_episodes:
            visual_input = torch.tensor(episode['visual']).unsqueeze(0).to(self.device)
            audio_input = torch.tensor(episode['audio']).unsqueeze(0).to(self.device)
            text_input = torch.tensor(episode['text']).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(visual_input, audio_input, text_input)
            
            # Simulate user satisfaction based on personalization
            retrieved_memories = outputs['retrieved_memories']
            if len(retrieved_memories) > 0:
                # Higher memory retrieval indicates better personalization
                memory_quality = min(len(retrieved_memories) / 10.0, 1.0)
                satisfaction = 6.0 + 4.0 * memory_quality  # Scale to 6-10 range
            else:
                satisfaction = 6.0  # Baseline satisfaction
            
            satisfaction_scores.append(satisfaction)
        
        avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores)
        return {'satisfaction': avg_satisfaction, 'total_episodes': len(assist_episodes)}
    
    def evaluate_lab_assist(self, lab_episodes: List[Dict]) -> Dict:
        """Evaluate on LabAssist benchmark task."""
        successful_assists = 0
        total_assists = len(lab_episodes)
        
        for episode in lab_episodes:
            visual_input = torch.tensor(episode['visual']).unsqueeze(0).to(self.device)
            audio_input = torch.tensor(episode['audio']).unsqueeze(0).to(self.device)
            text_input = torch.tensor(episode['text']).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(visual_input, audio_input, text_input)
            
            # Simulate lab assistance success
            retrieved_memories = outputs['retrieved_memories']
            episode_embedding = outputs['current_episode']['episode_embedding']
            
            # Use embedding quality and memory retrieval for success prediction
            embedding_quality = torch.norm(episode_embedding).item()
            memory_support = len(retrieved_memories) > 0
            
            if embedding_quality > 10.0 and memory_support:  # Arbitrary thresholds
                successful_assists += 1
        
        success_rate = successful_assists / total_assists
        return {'success_rate': success_rate, 'total_episodes': total_assists}


def main():
    """Main training and evaluation script."""
    # Configuration
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'max_epochs': 50,
        'batch_size': 32,
        'alignment_weight': 0.1,
        'task_weight': 1.0,
        'visual_dim': 512,
        'audio_dim': 256,
        'text_dim': 768,
        'fusion_dim': 1024,
        'num_heads': 8,
        'num_layers': 6,
        'hot_capacity': 1000,
        'warm_capacity': 5000,
        'cold_capacity': 50000
    }
    
    # Initialize model
    model = TMMB(
        visual_dim=config['visual_dim'],
        audio_dim=config['audio_dim'],
        text_dim=config['text_dim'],
        fusion_dim=config['fusion_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        hot_capacity=config['hot_capacity'],
        warm_capacity=config['warm_capacity'],
        cold_capacity=config['cold_capacity']
    )
    
    # Create datasets
    train_dataset = MultimodalDataset('data/train')
    val_dataset = MultimodalDataset('data/val')
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False
    )
    
    # Initialize trainer
    trainer = TMBBTrainer(model, config)
    
    # Train model
    logger.info("Starting training...")
    trainer.train(train_dataloader, val_dataloader, config['max_epochs'])
    
    # Benchmark evaluation
    logger.info("Starting benchmark evaluation...")
    evaluator = BenchmarkEvaluator(model)
    
    # Generate dummy benchmark data for demonstration
    nav_episodes = [{'visual': np.random.randn(2048), 'audio': np.random.randn(768), 'text': np.random.randn(768)} for _ in range(100)]
    assist_episodes = [{'visual': np.random.randn(2048), 'audio': np.random.randn(768), 'text': np.random.randn(768)} for _ in range(50)]
    lab_episodes = [{'visual': np.random.randn(2048), 'audio': np.random.randn(768), 'text': np.random.randn(768)} for _ in range(200)]
    
    # Evaluate benchmarks
    multinav_results = evaluator.evaluate_multinav(nav_episodes)
    personal_assist_results = evaluator.evaluate_personal_assist(assist_episodes)  
    lab_assist_results = evaluator.evaluate_lab_assist(lab_episodes)
    
    # Print results
    logger.info("Benchmark Results:")
    logger.info(f"MultiNav Success Rate: {multinav_results['success_rate']:.1%}")
    logger.info(f"PersonalAssist Satisfaction: {personal_assist_results['satisfaction']:.1f}/10")
    logger.info(f"LabAssist Success Rate: {lab_assist_results['success_rate']:.1%}")
    
    # Save final results
    results = {
        'multinav': multinav_results,
        'personal_assist': personal_assist_results,
        'lab_assist': lab_assist_results,
        'memory_stats': model.get_memory_stats()
    }
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Training and evaluation completed!")

if __name__ == "__main__":
    main()
