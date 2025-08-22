"""
Hallucination Detector for RAG Assistant

This module provides functionality to detect potential hallucinations in RAG responses
by analyzing the relationship between the query, retrieved citations, and generated answer.
"""

from typing import Dict, List, Any, Tuple
import re
from difflib import SequenceMatcher
try:
    import numpy as np
except ImportError:
    print("Warning: numpy not available, using fallback calculations")
    np = None


class HallucinationDetector:
    """Detects potential hallucinations in RAG responses."""
    
    def __init__(self):
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
    
    def detect_hallucination(
        self, 
        query: str, 
        answer: str, 
        citations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect potential hallucinations in the RAG response.
        
        Args:
            query: The user's original query
            answer: The generated answer
            citations: List of citation dictionaries with 'text' and 'score' keys
            
        Returns:
            Dictionary containing hallucination analysis results
        """
        try:
            print(f"Starting hallucination detection for query: {query[:50]}...")
            
            # Handle None answer
            if answer is None:
                answer = ""
            
            print(f"Answer length: {len(answer)}")
            print(f"Number of citations: {len(citations)}")
            
            # Check if this is an error response or failure message
            error_indicators = [
                "i couldn't generate a response",
                "i apologize, but",
                "no response generated",
                "error occurred",
                "failed to generate",
                "please try again"
            ]
            
            # Check if this is a legitimate "no information found" response
            no_info_indicators = [
                "no information",
                "not found",
                "doesn't exist",
                "not available",
                "no data",
                "not in the provided",
                "not in the citations"
            ]
            
            answer_lower = answer.lower()
            is_error_response = any(indicator in answer_lower for indicator in error_indicators)
            is_no_info_response = any(indicator in answer_lower for indicator in no_info_indicators)
            
            if is_error_response:
                print("Detected error response - skipping hallucination analysis")
                return {
                    'is_hallucination': False,
                    'confidence': 'low',
                    'score': 0.0,
                    'reason': 'Response generation failed - no hallucination analysis possible',
                    'suggestions': ['Try again with a different query', 'Check API configuration'],
                    'metrics': {
                        'citation_coverage': 0.0,
                        'query_relevance': 0.0,
                        'citation_quality': 0.0
                    }
                }
            
            if is_no_info_response and citations:
                print("Detected legitimate 'no information found' response")
                return {
                    'is_hallucination': False,
                    'confidence': 'high',
                    'score': 0.0,
                    'reason': 'Legitimate response indicating requested information not found in available data',
                    'suggestions': ['Try a different query', 'Check if the requested data exists in the dataset'],
                    'metrics': {
                        'citation_coverage': 0.0,
                        'query_relevance': 0.5,  # Query was relevant, just no matching data
                        'citation_quality': 0.5
                    }
                }
            
            if not citations:
                print("No citations provided")
                return {
                    'is_hallucination': True,
                    'confidence': 'high',
                    'score': 1.0,
                    'reason': 'No citations provided - answer not grounded in data',
                    'suggestions': ['Upload more relevant data', 'Try a different query'],
                    'metrics': {
                        'citation_coverage': 0.0,
                        'query_relevance': 0.0,
                        'citation_quality': 0.0
                    }
                }
            
            # Calculate various metrics
            print("Calculating citation coverage...")
            citation_coverage = self._calculate_citation_coverage(answer, citations)
            print(f"Citation coverage: {citation_coverage}")
            
            print("Calculating query relevance...")
            query_relevance = self._calculate_query_relevance(query, answer)
            print(f"Query relevance: {query_relevance}")
            
            print("Calculating citation quality...")
            citation_quality = self._calculate_citation_quality(citations)
            print(f"Citation quality: {citation_quality}")
            
            print(f"Metrics calculated - Coverage: {citation_coverage:.3f}, Relevance: {query_relevance:.3f}, Quality: {citation_quality:.3f}")
            
            # Combined hallucination score (0 = no hallucination, 1 = likely hallucination)
            hallucination_score = 1.0 - (citation_coverage * 0.5 + query_relevance * 0.3 + citation_quality * 0.2)
            
            # Determine confidence level
            confidence = self._get_confidence_level(hallucination_score)
            
            # Determine if it's likely a hallucination
            is_hallucination = hallucination_score > self.confidence_thresholds['medium']
            
            # Generate reason and suggestions
            reason, suggestions = self._generate_feedback(
                hallucination_score, citation_coverage, query_relevance, citation_quality
            )
            
            result = {
                'is_hallucination': is_hallucination,
                'confidence': confidence,
                'score': round(hallucination_score, 3),
                'metrics': {
                    'citation_coverage': round(citation_coverage, 3),
                    'query_relevance': round(query_relevance, 3),
                    'citation_quality': round(citation_quality, 3)
                },
                'reason': reason,
                'suggestions': suggestions
            }
            
            print(f"Hallucination detection completed - Score: {hallucination_score:.3f}, Is hallucination: {is_hallucination}")
            return result
            
        except Exception as e:
            print(f"Error in hallucination detection: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            # Return a safe fallback
            return {
                'is_hallucination': False,
                'confidence': 'low',
                'score': 0.5,
                'reason': f'Error in hallucination detection: {str(e)} - using fallback',
                'suggestions': ['Try the query again'],
                'metrics': {
                    'citation_coverage': 0.0,
                    'query_relevance': 0.0,
                    'citation_quality': 0.0
                }
            }
    
    def _calculate_citation_coverage(self, answer: str, citations: List[Dict[str, Any]]) -> float:
        """Calculate how much of the answer is covered by citations."""
        try:
            if not citations:
                return 0.0
            
            # Extract key phrases from answer (simple approach)
            try:
                answer_phrases = self._extract_key_phrases(answer.lower())
            except Exception as e:
                print(f"Error extracting phrases from answer: {e}")
                answer_phrases = []
            
            # Extract key phrases from citations
            citation_phrases = set()
            for citation in citations:
                try:
                    # Handle different possible citation structures
                    citation_text = citation.get('text', citation.get('content', citation.get('sample_post', '')))
                    if citation_text:
                        citation_text = str(citation_text).lower()
                        citation_phrases.update(self._extract_key_phrases(citation_text))
                except Exception as e:
                    print(f"Error processing citation: {e}")
                    continue
            
            if not answer_phrases:
                return 0.0
            
            # Calculate coverage
            covered_phrases = 0
            for phrase in answer_phrases:
                try:
                    if any(self._phrase_similarity(phrase, cit_phrase) > 0.7 for cit_phrase in citation_phrases):
                        covered_phrases += 1
                except Exception as e:
                    print(f"Error calculating phrase similarity: {e}")
                    continue
            
            coverage = covered_phrases / len(answer_phrases) if answer_phrases else 0.0
            return coverage
            
        except Exception as e:
            print(f"Error in citation coverage calculation: {e}")
            return 0.0
    
    def _calculate_query_relevance(self, query: str, answer: str) -> float:
        """Calculate how relevant the answer is to the query."""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        
        if not query_words:
            return 0.0
        
        # Calculate word overlap
        overlap = len(query_words.intersection(answer_words))
        basic_relevance = overlap / len(query_words)
        
        # Special handling for specific query types
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Handle handle-specific queries
        if "@" in query and "handle" in query_lower:
            # Extract handle from query
            handle_match = re.search(r'@(\w+)', query)
            if handle_match:
                target_handle = handle_match.group(1)
                if target_handle.lower() in answer_lower:
                    return 1.0  # Perfect relevance if handle is mentioned in answer
                elif "not found" in answer_lower or "no information" in answer_lower:
                    return 0.8  # High relevance for legitimate "not found" responses
        
        # Handle niche queries
        if "niche" in query_lower:
            if "niche" in answer_lower:
                return max(basic_relevance, 0.9)  # Boost relevance for niche-related answers
        
        # Handle "who talks about" queries
        if "who" in query_lower and ("talks" in query_lower or "discuss" in query_lower):
            if any(word in answer_lower for word in ["influencer", "handle", "@", "talks", "discuss"]):
                return max(basic_relevance, 0.8)
        
        # Handle content queries
        if any(word in query_lower for word in ["content", "post", "tweet", "message"]):
            if any(word in answer_lower for word in ["post", "tweet", "content", "message"]):
                return max(basic_relevance, 0.8)
        
        return basic_relevance
    
    def _calculate_citation_quality(self, citations: List[Dict[str, Any]]) -> float:
        """Calculate the quality of citations based on scores and content."""
        if not citations:
            return 0.0
        
        # Average citation score
        scores = [citation.get('score', 0) for citation in citations]
        if np is not None:
            avg_score = np.mean(scores) if scores else 0.0
        else:
            avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Citation diversity (more diverse citations = better)
        citation_texts = []
        for citation in citations:
            # Handle different possible citation structures
            text = citation.get('text', citation.get('content', citation.get('sample_post', '')))
            citation_texts.append(text)
        
        diversity = len(set(citation_texts)) / len(citation_texts) if citation_texts else 0.0
        
        # Metadata completeness (check if citations have useful metadata)
        metadata_completeness = 0.0
        for citation in citations:
            metadata = citation.get("metadata", {})
            name = metadata.get("name", citation.get("name", ""))
            handle = metadata.get("handle", citation.get("handle", ""))
            niche = metadata.get("niche", citation.get("niche", ""))
            post = metadata.get("sample_post", citation.get("sample_post", ""))
            
            # Score completeness (0-1)
            completeness = 0.0
            if name: completeness += 0.25
            if handle: completeness += 0.25
            if niche: completeness += 0.25
            if post: completeness += 0.25
            metadata_completeness += completeness
        
        metadata_completeness = metadata_completeness / len(citations) if citations else 0.0
        
        # Combine metrics with weights
        quality_score = (
            avg_score * 0.4 +           # 40% weight to relevance score
            diversity * 0.3 +           # 30% weight to content diversity
            metadata_completeness * 0.3  # 30% weight to metadata completeness
        )
        
        return quality_score
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        try:
            if not text or not isinstance(text, str):
                return []
            
            # Simple approach: extract noun phrases and important words
            words = re.findall(r'\b\w+\b', text)
            if not words:
                return []
                
            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            key_words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
            
            if not key_words:
                return []
            
            # Create phrases from consecutive words
            phrases = []
            for i in range(len(key_words) - 1):
                phrases.append(f"{key_words[i]} {key_words[i+1]}")
            
            return key_words + phrases
            
        except Exception as e:
            print(f"Error in _extract_key_phrases: {e}")
            return []
    
    def _phrase_similarity(self, phrase1: str, phrase2: str) -> float:
        """Calculate similarity between two phrases."""
        return SequenceMatcher(None, phrase1, phrase2).ratio()
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level based on hallucination score."""
        if score >= self.confidence_thresholds['high']:
            return 'high'
        elif score >= self.confidence_thresholds['medium']:
            return 'medium'
        elif score >= self.confidence_thresholds['low']:
            return 'low'
        else:
            return 'very_low'
    
    def _generate_feedback(self, hallucination_score: float, citation_coverage: float, 
                          query_relevance: float, citation_quality: float) -> Tuple[str, List[str]]:
        """Generate feedback based on analysis results."""
        reasons = []
        suggestions = []
        
        if citation_coverage < 0.3:
            reasons.append("Low citation coverage - answer may contain unsupported claims")
            suggestions.append("Ask for more specific information from the dataset")
        
        if query_relevance < 0.4:
            reasons.append("Answer may not directly address the query")
            suggestions.append("Rephrase your question to be more specific")
        
        if citation_quality < 0.5:
            reasons.append("Low-quality citations - consider uploading more relevant data")
            suggestions.append("Upload additional relevant datasets")
        
        if hallucination_score > 0.7:
            reasons.append("High likelihood of hallucination detected")
            suggestions.append("Verify information with original sources")
        
        if not reasons:
            reasons.append("Response appears to be well-grounded in the provided data")
            suggestions.append("Continue with confidence")
        
        return "; ".join(reasons), suggestions


# Global instance
hallucination_detector = HallucinationDetector()
