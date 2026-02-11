"""
AI Agents for Brain MRI Tumor Classification.

Architecture:
============

OFFLINE PHASE (Analysis-time)
-----------------------------
Agent 1: Model Performance Evaluation Agent
  - Consumes: metrics, CV results, error analysis, inference times
  - Outputs: Model reliability profiles (STORED for reuse)

ONLINE PHASE (Inference-time)  
-----------------------------
Agent 2: Prediction Consensus Agent
  - Inputs: Live predictions + Agent 1's reliability scores
  - Outputs: Weighted consensus prediction

Agent 3: Risk & Uncertainty Assessment Agent
  - Inputs: Agent 2's output + confidence spread
  - Outputs: Risk level + recommendation


Data Flow:
=========

(Training & Testing)
  ├─ Metrics (Accuracy, Precision, Recall, F1)
  ├─ Cross-validation (mean ± std)
  ├─ Error analysis (confusion patterns)
  ├─ Inference time
  ↓
Agent 1: Performance Evaluation
  ↓ (stored model reliability profiles)
--------------------------------
(New MRI Image)
  ↓
CNN models predict
  ↓
Agent 2: Prediction Consensus (uses Agent 1's scores)
  ↓
Agent 3: Risk & Uncertainty
  ↓
Final output (class + confidence + recommendation)
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import warnings

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, MODELS, CLASSES, NUM_CLASSES


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class ModelMetrics:
    """Metrics for a single model (input to Agent 1)."""
    model_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Cross-validation results
    cv_accuracy_mean: float = 0.0
    cv_accuracy_std: float = 0.0
    cv_f1_mean: float = 0.0
    cv_f1_std: float = 0.0
    
    # Class-wise performance
    class_f1_scores: Dict[str, float] = field(default_factory=dict)
    class_recall_scores: Dict[str, float] = field(default_factory=dict)
    
    # Error analysis
    total_errors: int = 0
    high_confidence_errors: int = 0
    most_confused_classes: List[Tuple[str, str, int]] = field(default_factory=list)
    
    # Inference performance
    inference_time_ms: float = 0.0
    
    # Calibration
    ece: float = 0.0  # Expected Calibration Error


@dataclass
class ModelReliabilityProfile:
    """Output of Agent 1 - stored for use by Agent 2."""
    model_name: str
    
    # Overall reliability score (0-1)
    reliability_score: float = 0.0
    
    # Component scores
    accuracy_score: float = 0.0
    stability_score: float = 0.0  # From CV variance
    calibration_score: float = 0.0
    speed_score: float = 0.0
    
    # Qualitative assessment
    tier: str = ""  # TOP, HIGH, RELIABLE, WEAK
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    
    # Class-specific reliability
    class_reliability: Dict[str, float] = field(default_factory=dict)
    weak_classes: List[str] = field(default_factory=list)
    
    # Voting weight for consensus
    voting_weight: float = 0.0
    
    # Metadata
    evaluated_at: str = ""


@dataclass
class ModelPrediction:
    """Single model's prediction for an image."""
    model_name: str
    predicted_class: str
    predicted_class_idx: int
    confidence: float
    class_probabilities: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConsensusPrediction:
    """Output of Agent 2."""
    final_class: str
    final_class_idx: int
    consensus_confidence: float
    
    # Voting details
    agreement_count: int = 0
    total_models: int = 0
    agreement_ratio: float = 0.0
    
    # Weighted voting result
    weighted_class_scores: Dict[str, float] = field(default_factory=dict)
    
    # Individual predictions
    individual_predictions: Dict[str, str] = field(default_factory=dict)
    
    # Consensus strength
    consensus_level: str = ""  # HIGH, MEDIUM, LOW


class RiskLevel(Enum):
    """Risk levels for Agent 3 output."""
    HIGH_CONFIDENCE = "high_confidence"
    MODERATE = "moderate"
    UNCERTAIN = "uncertain"
    REVIEW_REQUIRED = "review_required"


@dataclass
class RiskAssessment:
    """Output of Agent 3."""
    risk_level: RiskLevel
    risk_score: float  # 0-1, higher = more uncertain
    
    # Factors
    confidence_variance: float = 0.0
    model_disagreement: int = 0
    
    # Flags
    flags: List[str] = field(default_factory=list)
    
    # Recommendation
    recommendation: str = ""
    clinical_note: str = ""


@dataclass
class DiagnosticOutput:
    """Final combined output from all agents."""
    # From Agent 2
    predicted_class: str
    confidence: float
    consensus_level: str
    
    # From Agent 3
    risk_level: str
    recommendation: str
    
    # Details
    model_predictions: Dict[str, Dict] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: str = ""


# ==============================================================================
# AGENT 1: MODEL PERFORMANCE EVALUATION AGENT (OFFLINE)
# ==============================================================================

class ModelPerformanceAgent:
    """
    OFFLINE PHASE: Model Performance Evaluation Agent
    
    This agent does NOT look at images.
    It analyzes metrics, CV results, and error patterns to create
    model reliability profiles that are STORED for later use.
    
    Inputs:
        - Accuracy, Precision, Recall, F1-score
        - Cross-validation results (mean ± std)
        - Error analysis findings
        - Inference time
        - Calibration metrics
    
    Outputs:
        - ModelReliabilityProfile for each model (stored to disk)
    """
    
    # Weights for computing reliability score
    WEIGHTS = {
        "f1_score": 0.35,       # Primary metric
        "recall": 0.25,         # Critical for medical (catch all cases)
        "precision": 0.15,      # Avoid false positives
        "stability": 0.15,      # CV variance
        "calibration": 0.05,    # Confidence quality
        "speed": 0.05           # Deployment consideration
    }
    
    # Thresholds for tier assignment
    TIER_THRESHOLDS = {
        "TOP": 0.90,
        "HIGH": 0.80,
        "RELIABLE": 0.70,
        "WEAK": 0.0
    }
    
    def __init__(self, profiles_path: Optional[Path] = None):
        """
        Initialize Agent 1.
        
        Args:
            profiles_path: Path to store/load reliability profiles
        """
        self.profiles_path = profiles_path or RESULTS_DIR / "model_reliability_profiles.json"
        self.profiles: Dict[str, ModelReliabilityProfile] = {}
        
        # Try to load existing profiles
        self._load_profiles()
    
    def _load_profiles(self) -> None:
        """Load existing profiles from disk."""
        if self.profiles_path.exists():
            try:
                with open(self.profiles_path, 'r') as f:
                    data = json.load(f)
                
                for name, profile_dict in data.items():
                    self.profiles[name] = ModelReliabilityProfile(**profile_dict)
                
                print(f"Loaded {len(self.profiles)} model profiles from disk")
            except Exception as e:
                print(f"Could not load profiles: {e}")
    
    def _save_profiles(self) -> None:
        """Save profiles to disk."""
        data = {name: asdict(profile) for name, profile in self.profiles.items()}
        
        with open(self.profiles_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.profiles)} model profiles to {self.profiles_path}")
    
    def evaluate_model(self, metrics: ModelMetrics) -> ModelReliabilityProfile:
        """
        Evaluate a single model and create its reliability profile.
        
        Args:
            metrics: ModelMetrics with all evaluation data
            
        Returns:
            ModelReliabilityProfile
        """
        profile = ModelReliabilityProfile(model_name=metrics.model_name)
        
        # 1. Compute component scores
        profile.accuracy_score = metrics.f1_score
        
        # Stability score (lower CV std = higher stability)
        if metrics.cv_f1_std > 0:
            profile.stability_score = max(0, 1 - metrics.cv_f1_std * 10)
        else:
            profile.stability_score = 0.8  # Default if no CV
        
        # Calibration score (lower ECE = better calibration)
        profile.calibration_score = max(0, 1 - metrics.ece * 10)
        
        # Speed score (normalize: assume 200ms max)
        profile.speed_score = max(0, 1 - metrics.inference_time_ms / 200)
        
        # 2. Compute overall reliability score
        profile.reliability_score = (
            self.WEIGHTS["f1_score"] * metrics.f1_score +
            self.WEIGHTS["recall"] * metrics.recall +
            self.WEIGHTS["precision"] * metrics.precision +
            self.WEIGHTS["stability"] * profile.stability_score +
            self.WEIGHTS["calibration"] * profile.calibration_score +
            self.WEIGHTS["speed"] * profile.speed_score
        )
        
        # 3. Assign tier
        for tier, threshold in self.TIER_THRESHOLDS.items():
            if profile.reliability_score >= threshold:
                profile.tier = tier
                break
        
        # 4. Identify strengths and weaknesses
        profile.strengths, profile.weaknesses = self._analyze_strengths_weaknesses(metrics)
        
        # 5. Class-specific reliability
        for cls, f1 in metrics.class_f1_scores.items():
            profile.class_reliability[cls] = f1
            if f1 < 0.80:  # Threshold for weak class
                profile.weak_classes.append(cls)
        
        # 6. Compute voting weight (used by Agent 2)
        # Weight based on reliability, but ensure minimum participation
        profile.voting_weight = max(0.1, profile.reliability_score)
        
        # 7. Metadata
        profile.evaluated_at = datetime.now().isoformat()
        
        # Store profile
        self.profiles[metrics.model_name] = profile
        
        return profile
    
    def _analyze_strengths_weaknesses(
        self, 
        metrics: ModelMetrics
    ) -> Tuple[List[str], List[str]]:
        """Identify model strengths and weaknesses."""
        strengths = []
        weaknesses = []
        
        # Accuracy-based
        if metrics.f1_score >= 0.95:
            strengths.append("Excellent F1 score (>=95%)")
        elif metrics.f1_score < 0.80:
            weaknesses.append("Low F1 score (<80%)")
        
        if metrics.recall >= 0.95:
            strengths.append("High recall - catches most cases")
        elif metrics.recall < 0.85:
            weaknesses.append("Lower recall - may miss cases")
        
        # Stability
        if metrics.cv_f1_std < 0.02:
            strengths.append("Stable across cross-validation folds")
        elif metrics.cv_f1_std > 0.05:
            weaknesses.append("High variance across folds")
        
        # Calibration
        if metrics.ece < 0.05:
            strengths.append("Well-calibrated confidence scores")
        elif metrics.ece > 0.10:
            weaknesses.append("Poorly calibrated - confidence unreliable")
        
        # Speed
        if metrics.inference_time_ms < 30:
            strengths.append("Fast inference (real-time capable)")
        elif metrics.inference_time_ms > 100:
            weaknesses.append("Slow inference")
        
        # Error patterns
        if metrics.high_confidence_errors > 5:
            weaknesses.append(f"High-confidence errors ({metrics.high_confidence_errors})")
        
        # Class-specific
        for cls, recall in metrics.class_recall_scores.items():
            if recall < 0.75:
                weaknesses.append(f"Weak on {cls} class (recall: {recall:.0%})")
        
        return strengths, weaknesses
    
    def evaluate_all_models(self, metrics_list: List[ModelMetrics]) -> Dict[str, ModelReliabilityProfile]:
        """
        Evaluate all models and store profiles.
        
        Args:
            metrics_list: List of ModelMetrics for each model
            
        Returns:
            Dictionary of model name -> reliability profile
        """
        print("\n" + "=" * 60)
        print("AGENT 1: MODEL PERFORMANCE EVALUATION")
        print("=" * 60)
        
        for metrics in metrics_list:
            print(f"\nEvaluating: {metrics.model_name}")
            profile = self.evaluate_model(metrics)
            print(f"  Reliability Score: {profile.reliability_score:.3f}")
            print(f"  Tier: {profile.tier}")
            print(f"  Voting Weight: {profile.voting_weight:.3f}")
        
        # Save all profiles
        self._save_profiles()
        
        return self.profiles
    
    def get_profile(self, model_name: str) -> Optional[ModelReliabilityProfile]:
        """Get reliability profile for a model."""
        return self.profiles.get(model_name)
    
    def get_all_profiles(self) -> Dict[str, ModelReliabilityProfile]:
        """Get all reliability profiles."""
        return self.profiles
    
    def generate_report(self) -> str:
        """Generate evaluation report for all models."""
        lines = []
        lines.append("=" * 70)
        lines.append("MODEL RELIABILITY PROFILES (Agent 1 Output)")
        lines.append("=" * 70)
        
        # Sort by reliability score
        sorted_profiles = sorted(
            self.profiles.items(),
            key=lambda x: x[1].reliability_score,
            reverse=True
        )
        
        for rank, (name, profile) in enumerate(sorted_profiles, 1):
            tier_emoji = {
                "TOP": "[TOP]",
                "HIGH": "[HIGH]",
                "RELIABLE": "[OK]",
                "WEAK": "[!]"
            }.get(profile.tier, "")
            
            lines.append(f"\n#{rank} {name.upper()} {tier_emoji}")
            lines.append("-" * 50)
            lines.append(f"Reliability Score: {profile.reliability_score:.3f}")
            lines.append(f"Tier: {profile.tier}")
            lines.append(f"Voting Weight: {profile.voting_weight:.3f}")
            
            if profile.strengths:
                lines.append(f"\nStrengths:")
                for s in profile.strengths:
                    lines.append(f"  + {s}")
            
            if profile.weaknesses:
                lines.append(f"\nWeaknesses:")
                for w in profile.weaknesses:
                    lines.append(f"  - {w}")
            
            if profile.weak_classes:
                lines.append(f"\nWeak Classes: {', '.join(profile.weak_classes)}")
        
        lines.append("\n" + "=" * 70)
        lines.append(f"Profiles stored at: {self.profiles_path}")
        lines.append("=" * 70)
        
        return "\n".join(lines)


# ==============================================================================
# AGENT 2: PREDICTION CONSENSUS AGENT (ONLINE)
# ==============================================================================

class PredictionConsensusAgent:
    """
    ONLINE PHASE: Prediction Consensus Agent
    
    Uses LIVE predictions weighted by Agent 1's reliability scores.
    
    Inputs:
        - ModelPrediction from each model (for current image)
        - ModelReliabilityProfile from Agent 1 (stored)
    
    Outputs:
        - ConsensusPrediction with final class + confidence
    """
    
    def __init__(self, performance_agent: ModelPerformanceAgent):
        """
        Initialize Agent 2.
        
        Args:
            performance_agent: Reference to Agent 1 for reliability profiles
        """
        self.performance_agent = performance_agent
    
    def compute_consensus(
        self, 
        predictions: List[ModelPrediction]
    ) -> ConsensusPrediction:
        """
        Compute weighted consensus from multiple model predictions.
        
        Logic:
        1. Get reliability scores from Agent 1
        2. Weight each model's vote by its reliability
        3. Compute weighted class scores
        4. Determine consensus and confidence
        
        Args:
            predictions: List of ModelPrediction from each model
            
        Returns:
            ConsensusPrediction
        """
        if not predictions:
            raise ValueError("No predictions provided")
        
        # Initialize weighted scores for each class
        weighted_scores = {cls: 0.0 for cls in CLASSES}
        total_weight = 0.0
        
        # Track individual predictions
        individual_preds = {}
        
        for pred in predictions:
            # Get model's reliability profile from Agent 1
            profile = self.performance_agent.get_profile(pred.model_name)
            
            if profile is not None:
                weight = profile.voting_weight
            else:
                weight = 0.5  # Default weight if no profile
                warnings.warn(f"No reliability profile for {pred.model_name}, using default weight")
            
            # Add weighted vote
            # Vote = confidence × reliability weight
            for cls, prob in pred.class_probabilities.items():
                weighted_scores[cls] += prob * weight
            
            total_weight += weight
            individual_preds[pred.model_name] = pred.predicted_class
        
        # Normalize scores
        if total_weight > 0:
            weighted_scores = {cls: score / total_weight for cls, score in weighted_scores.items()}
        
        # Determine final prediction
        final_class = max(weighted_scores, key=weighted_scores.get)
        final_class_idx = CLASSES.index(final_class)
        consensus_confidence = weighted_scores[final_class]
        
        # Calculate agreement
        predictions_list = list(individual_preds.values())
        agreement_count = sum(1 for p in predictions_list if p == final_class)
        total_models = len(predictions)
        agreement_ratio = agreement_count / total_models
        
        # Determine consensus level
        if agreement_ratio >= 0.8 and consensus_confidence >= 0.8:
            consensus_level = "HIGH"
        elif agreement_ratio >= 0.6 and consensus_confidence >= 0.6:
            consensus_level = "MEDIUM"
        else:
            consensus_level = "LOW"
        
        return ConsensusPrediction(
            final_class=final_class,
            final_class_idx=final_class_idx,
            consensus_confidence=consensus_confidence,
            agreement_count=agreement_count,
            total_models=total_models,
            agreement_ratio=agreement_ratio,
            weighted_class_scores=weighted_scores,
            individual_predictions=individual_preds,
            consensus_level=consensus_level
        )


# ==============================================================================
# AGENT 3: RISK & UNCERTAINTY ASSESSMENT AGENT (ONLINE)
# ==============================================================================

class RiskAssessmentAgent:
    """
    ONLINE PHASE: Risk & Uncertainty Assessment Agent
    
    Safety layer that knows when NOT to be confident.
    
    Inputs:
        - ConsensusPrediction from Agent 2
        - Individual model predictions (for variance analysis)
    
    Outputs:
        - RiskAssessment with risk level + recommendation
    """
    
    # Risk thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    MODERATE_THRESHOLD = 0.65
    UNCERTAIN_THRESHOLD = 0.50
    
    AGREEMENT_HIGH = 0.80
    AGREEMENT_MODERATE = 0.60
    
    def __init__(self, performance_agent: ModelPerformanceAgent):
        """
        Initialize Agent 3.
        
        Args:
            performance_agent: Reference to Agent 1 for class-specific reliability
        """
        self.performance_agent = performance_agent
    
    def assess_risk(
        self,
        consensus: ConsensusPrediction,
        predictions: List[ModelPrediction]
    ) -> RiskAssessment:
        """
        Assess risk and uncertainty of the prediction.
        
        Args:
            consensus: Output from Agent 2
            predictions: Original predictions for variance analysis
            
        Returns:
            RiskAssessment
        """
        flags = []
        
        # 1. Calculate confidence variance across models
        confidences = [p.confidence for p in predictions]
        confidence_variance = np.var(confidences)
        confidence_mean = np.mean(confidences)
        
        # 2. Count model disagreement
        unique_predictions = set(p.predicted_class for p in predictions)
        model_disagreement = len(unique_predictions) - 1  # 0 = all agree
        
        # 3. Check for concerning patterns
        
        # Low overall confidence
        if consensus.consensus_confidence < self.UNCERTAIN_THRESHOLD:
            flags.append("LOW_CONFIDENCE")
        
        # High variance in confidences (models are uncertain)
        if confidence_variance > 0.05:
            flags.append("HIGH_CONFIDENCE_VARIANCE")
        
        # Significant disagreement
        if model_disagreement >= 2:
            flags.append("SIGNIFICANT_DISAGREEMENT")
        elif model_disagreement >= 1:
            flags.append("SOME_DISAGREEMENT")
        
        # Close second-place prediction
        sorted_scores = sorted(consensus.weighted_class_scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            margin = sorted_scores[0] - sorted_scores[1]
            if margin < 0.15:
                flags.append("CLOSE_RUNNER_UP")
        
        # Check if predicted class is a known weak spot
        for profile in self.performance_agent.get_all_profiles().values():
            if consensus.final_class in profile.weak_classes:
                flags.append("WEAK_CLASS_PREDICTED")
                break
        
        # 4. Determine risk level
        risk_level, risk_score = self._determine_risk_level(
            consensus, confidence_variance, model_disagreement, flags
        )
        
        # 5. Generate recommendation
        recommendation, clinical_note = self._generate_recommendation(
            risk_level, consensus, flags
        )
        
        return RiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            confidence_variance=confidence_variance,
            model_disagreement=model_disagreement,
            flags=flags,
            recommendation=recommendation,
            clinical_note=clinical_note
        )
    
    def _determine_risk_level(
        self,
        consensus: ConsensusPrediction,
        confidence_variance: float,
        model_disagreement: int,
        flags: List[str]
    ) -> Tuple[RiskLevel, float]:
        """Determine risk level based on multiple factors."""
        
        # Count severity of flags
        severe_flags = ["LOW_CONFIDENCE", "SIGNIFICANT_DISAGREEMENT", "WEAK_CLASS_PREDICTED"]
        moderate_flags = ["SOME_DISAGREEMENT", "HIGH_CONFIDENCE_VARIANCE", "CLOSE_RUNNER_UP"]
        
        severe_count = sum(1 for f in flags if f in severe_flags)
        moderate_count = sum(1 for f in flags if f in moderate_flags)
        
        # Decision logic
        if severe_count == 0 and moderate_count == 0:
            if consensus.agreement_ratio >= self.AGREEMENT_HIGH and \
               consensus.consensus_confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
                return RiskLevel.HIGH_CONFIDENCE, 0.1
        
        if severe_count >= 2:
            return RiskLevel.REVIEW_REQUIRED, 0.9
        
        if severe_count >= 1 or moderate_count >= 2:
            return RiskLevel.UNCERTAIN, 0.7
        
        if moderate_count >= 1:
            return RiskLevel.MODERATE, 0.4
        
        return RiskLevel.HIGH_CONFIDENCE, 0.2
    
    def _generate_recommendation(
        self,
        risk_level: RiskLevel,
        consensus: ConsensusPrediction,
        flags: List[str]
    ) -> Tuple[str, str]:
        """Generate recommendation and clinical note."""
        
        if risk_level == RiskLevel.HIGH_CONFIDENCE:
            recommendation = "Prediction is reliable. Standard clinical workflow recommended."
            clinical_note = (
                f"AI prediction: {consensus.final_class} with high confidence "
                f"({consensus.consensus_confidence:.0%}). "
                f"{consensus.agreement_count}/{consensus.total_models} models agree."
            )
        
        elif risk_level == RiskLevel.MODERATE:
            recommendation = "Prediction is likely correct but review is suggested."
            clinical_note = (
                f"AI prediction: {consensus.final_class} with moderate confidence. "
                "Consider additional imaging or senior review."
            )
        
        elif risk_level == RiskLevel.UNCERTAIN:
            recommendation = "Prediction uncertain. Clinical review strongly recommended."
            reasons = []
            if "SIGNIFICANT_DISAGREEMENT" in flags:
                reasons.append("model disagreement")
            if "LOW_CONFIDENCE" in flags:
                reasons.append("low confidence")
            if "CLOSE_RUNNER_UP" in flags:
                reasons.append("close alternative diagnosis")
            
            clinical_note = (
                f"AI prediction: {consensus.final_class} (uncertain). "
                f"Uncertainty factors: {', '.join(reasons)}. "
                "Recommend radiologist review before diagnosis."
            )
        
        else:  # REVIEW_REQUIRED
            recommendation = "DO NOT use AI prediction alone. Mandatory clinical review."
            clinical_note = (
                f"AI system flagged high uncertainty. "
                f"Preliminary prediction: {consensus.final_class}. "
                "This case requires expert radiologist evaluation. "
                "Consider multi-disciplinary team review."
            )
        
        return recommendation, clinical_note


# ==============================================================================
# INTEGRATED PIPELINE
# ==============================================================================

class DiagnosticPipeline:
    """
    Complete diagnostic pipeline integrating all three agents.
    
    OFFLINE: Call evaluate_models() once with evaluation metrics
    ONLINE:  Call diagnose() for each new MRI image
    """
    
    def __init__(self, profiles_path: Optional[Path] = None):
        """
        Initialize the pipeline.
        
        Args:
            profiles_path: Path to store/load reliability profiles
        """
        # Agent 1: Performance Evaluation (OFFLINE)
        self.performance_agent = ModelPerformanceAgent(profiles_path)
        
        # Agent 2: Prediction Consensus (ONLINE)
        self.consensus_agent = PredictionConsensusAgent(self.performance_agent)
        
        # Agent 3: Risk Assessment (ONLINE)
        self.risk_agent = RiskAssessmentAgent(self.performance_agent)
    
    # ==================== OFFLINE PHASE ====================
    
    def evaluate_models(self, metrics_list: List[ModelMetrics]) -> None:
        """
        OFFLINE: Evaluate models and store reliability profiles.
        
        Call this ONCE after training/evaluation is complete.
        
        Args:
            metrics_list: List of ModelMetrics for each model
        """
        print("\n" + "=" * 70)
        print("OFFLINE PHASE: MODEL EVALUATION")
        print("=" * 70)
        
        self.performance_agent.evaluate_all_models(metrics_list)
        
        print(self.performance_agent.generate_report())
    
    def load_existing_profiles(self) -> bool:
        """Load existing profiles if available."""
        if self.performance_agent.profiles:
            print(f"Loaded {len(self.performance_agent.profiles)} model profiles")
            return True
        return False
    
    # ==================== ONLINE PHASE ====================
    
    def diagnose(self, predictions: List[ModelPrediction]) -> DiagnosticOutput:
        """
        ONLINE: Diagnose a single MRI image.
        
        Args:
            predictions: List of ModelPrediction from each model
            
        Returns:
            DiagnosticOutput with full diagnostic information
        """
        # Agent 2: Compute consensus
        consensus = self.consensus_agent.compute_consensus(predictions)
        
        # Agent 3: Assess risk
        risk = self.risk_agent.assess_risk(consensus, predictions)
        
        # Compile output
        output = DiagnosticOutput(
            predicted_class=consensus.final_class,
            confidence=consensus.consensus_confidence,
            consensus_level=consensus.consensus_level,
            risk_level=risk.risk_level.value,
            recommendation=risk.recommendation,
            model_predictions={
                p.model_name: {
                    "class": p.predicted_class,
                    "confidence": p.confidence
                }
                for p in predictions
            },
            flags=risk.flags,
            timestamp=datetime.now().isoformat()
        )
        
        return output
    
    def diagnose_with_report(self, predictions: List[ModelPrediction]) -> Tuple[DiagnosticOutput, str]:
        """
        Diagnose and generate human-readable report.
        
        Args:
            predictions: List of ModelPrediction from each model
            
        Returns:
            Tuple of (DiagnosticOutput, report_string)
        """
        output = self.diagnose(predictions)
        
        # Generate report
        lines = []
        lines.append("=" * 60)
        lines.append("DIAGNOSTIC REPORT")
        lines.append("=" * 60)
        
        # Main prediction
        lines.append(f"\n>> PREDICTION: {output.predicted_class.upper()}")
        lines.append(f"   Confidence: {output.confidence:.1%}")
        lines.append(f"   Consensus: {output.consensus_level}")
        
        # Risk assessment
        risk_emoji = {
            "high_confidence": "[OK]",
            "moderate": "[!]",
            "uncertain": "[?]",
            "review_required": "[X]"
        }.get(output.risk_level, "")
        
        lines.append(f"\n{risk_emoji} RISK LEVEL: {output.risk_level.upper()}")
        
        if output.flags:
            lines.append(f"   Flags: {', '.join(output.flags)}")
        
        lines.append(f"\n>> RECOMMENDATION:")
        lines.append(f"   {output.recommendation}")
        
        # Model breakdown
        lines.append(f"\n>> MODEL PREDICTIONS:")
        for model, pred_info in output.model_predictions.items():
            profile = self.performance_agent.get_profile(model)
            weight = profile.voting_weight if profile else 0.5
            lines.append(f"   {model}: {pred_info['class']} ({pred_info['confidence']:.1%}) [weight: {weight:.2f}]")
        
        lines.append("\n" + "=" * 60)
        
        report = "\n".join(lines)
        
        return output, report


# ==============================================================================
# DEMO / TESTING
# ==============================================================================

def demo_pipeline():
    """Demonstrate the complete pipeline."""
    
    print("\n" + "=" * 70)
    print("BRAIN MRI DIAGNOSTIC PIPELINE DEMO")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = DiagnosticPipeline()
    
    # ==================== OFFLINE PHASE ====================
    print("\n" + "-" * 70)
    print("PHASE 1: OFFLINE EVALUATION")
    print("-" * 70)
    
    # Simulated metrics from training/evaluation
    # 3-class tumor-type classification (no normal class)
    sample_metrics = [
        ModelMetrics(
            model_name="efficientnet_b4",
            accuracy=0.976, precision=0.971, recall=0.978, f1_score=0.974,
            cv_accuracy_mean=0.968, cv_accuracy_std=0.015,
            cv_f1_mean=0.965, cv_f1_std=0.018,
            class_f1_scores={"glioma": 0.96, "meningioma": 0.98, "pituitary": 0.99},
            class_recall_scores={"glioma": 0.97, "meningioma": 0.98, "pituitary": 0.99},
            total_errors=12, high_confidence_errors=2,
            inference_time_ms=45.2, ece=0.035
        ),
        ModelMetrics(
            model_name="resnet101",
            accuracy=0.963, precision=0.958, recall=0.961, f1_score=0.959,
            cv_accuracy_mean=0.955, cv_accuracy_std=0.022,
            cv_f1_mean=0.951, cv_f1_std=0.025,
            class_f1_scores={"glioma": 0.94, "meningioma": 0.97, "pituitary": 0.98},
            class_recall_scores={"glioma": 0.93, "meningioma": 0.96, "pituitary": 0.99},
            total_errors=18, high_confidence_errors=4,
            inference_time_ms=38.1, ece=0.048
        ),
        ModelMetrics(
            model_name="densenet121",
            accuracy=0.951, precision=0.946, recall=0.953, f1_score=0.949,
            cv_accuracy_mean=0.943, cv_accuracy_std=0.028,
            cv_f1_mean=0.938, cv_f1_std=0.031,
            class_f1_scores={"glioma": 0.92, "meningioma": 0.96, "pituitary": 0.97},
            class_recall_scores={"glioma": 0.91, "meningioma": 0.95, "pituitary": 0.98},
            total_errors=24, high_confidence_errors=5,
            inference_time_ms=32.5, ece=0.055
        ),
        ModelMetrics(
            model_name="resnet50",
            accuracy=0.945, precision=0.939, recall=0.947, f1_score=0.943,
            cv_accuracy_mean=0.938, cv_accuracy_std=0.025,
            cv_f1_mean=0.934, cv_f1_std=0.028,
            class_f1_scores={"glioma": 0.91, "meningioma": 0.95, "pituitary": 0.96},
            class_recall_scores={"glioma": 0.90, "meningioma": 0.94, "pituitary": 0.97},
            total_errors=27, high_confidence_errors=6,
            inference_time_ms=28.3, ece=0.062
        ),
        ModelMetrics(
            model_name="mobilenetv2",
            accuracy=0.921, precision=0.912, recall=0.925, f1_score=0.918,
            cv_accuracy_mean=0.912, cv_accuracy_std=0.032,
            cv_f1_mean=0.905, cv_f1_std=0.038,
            class_f1_scores={"glioma": 0.88, "meningioma": 0.93, "pituitary": 0.94},
            class_recall_scores={"glioma": 0.86, "meningioma": 0.92, "pituitary": 0.95},
            total_errors=35, high_confidence_errors=8,
            inference_time_ms=18.2, ece=0.078
        ),
    ]
    
    # Evaluate all models (stores profiles)
    pipeline.evaluate_models(sample_metrics)
    
    # ==================== ONLINE PHASE ====================
    print("\n" + "-" * 70)
    print("PHASE 2: ONLINE INFERENCE")
    print("-" * 70)
    
    # Case 1: High-confidence prediction (all models agree)
    print("\n[CASE 1] Clear Glioma Detection")
    predictions_case1 = [
        ModelPrediction("efficientnet_b4", "glioma", 0, 0.94, 
                       {"glioma": 0.94, "meningioma": 0.04, "pituitary": 0.02}),
        ModelPrediction("resnet101", "glioma", 0, 0.91,
                       {"glioma": 0.91, "meningioma": 0.06, "pituitary": 0.03}),
        ModelPrediction("densenet121", "glioma", 0, 0.89,
                       {"glioma": 0.89, "meningioma": 0.07, "pituitary": 0.04}),
        ModelPrediction("resnet50", "glioma", 0, 0.87,
                       {"glioma": 0.87, "meningioma": 0.08, "pituitary": 0.05}),
        ModelPrediction("mobilenetv2", "glioma", 0, 0.82,
                       {"glioma": 0.82, "meningioma": 0.11, "pituitary": 0.07}),
    ]
    
    output1, report1 = pipeline.diagnose_with_report(predictions_case1)
    print(report1)
    
    # Case 2: Uncertain prediction (models disagree)
    print("\n[CASE 2] Uncertain Case (Model Disagreement)")
    predictions_case2 = [
        ModelPrediction("efficientnet_b4", "meningioma", 1, 0.52,
                       {"glioma": 0.40, "meningioma": 0.52, "pituitary": 0.08}),
        ModelPrediction("resnet101", "glioma", 0, 0.48,
                       {"glioma": 0.48, "meningioma": 0.44, "pituitary": 0.08}),
        ModelPrediction("densenet121", "meningioma", 1, 0.51,
                       {"glioma": 0.42, "meningioma": 0.51, "pituitary": 0.07}),
        ModelPrediction("resnet50", "glioma", 0, 0.45,
                       {"glioma": 0.45, "meningioma": 0.46, "pituitary": 0.09}),
        ModelPrediction("mobilenetv2", "meningioma", 1, 0.47,
                       {"glioma": 0.44, "meningioma": 0.47, "pituitary": 0.09}),
    ]
    
    output2, report2 = pipeline.diagnose_with_report(predictions_case2)
    print(report2)
    
    print("\n[OK] Pipeline demonstration complete!")


if __name__ == "__main__":
    demo_pipeline()
