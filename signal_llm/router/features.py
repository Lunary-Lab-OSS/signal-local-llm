"""
Linguistic Feature Extraction

Extracts explicit difficulty signals (grade level, length, polysyllables)
to help the router detect complexity that semantic embeddings might miss.
"""

import logging

import textstat
import torch

logger = logging.getLogger(__name__)


class LinguisticFeatureExtractor:
    def __init__(self):
        # We normalize features based on typical question distributions
        # These constants should be calibrated on your specific dataset
        self.max_len = 50.0
        self.max_grade = 12.0

    def extract(self, text_list):
        """
        Extracts explicit difficulty signals:
        1. Flesch-Kincaid Grade Level
        2. Sentence Length (Word count)
        3. Polysyllable count
        """
        features = []
        for text in text_list:
            # CITATION [4]: Textstat Library for readability metrics
            try:
                grade = textstat.flesch_kincaid_grade(text)
                length = len(text.split())
                poly = textstat.polysyllabcount(text)

                # Normalize
                norm_grade = min(max(grade, 0), self.max_grade) / self.max_grade
                norm_len = min(length, self.max_len) / self.max_len
                norm_poly = min(poly, 10) / 10.0

                features.append([norm_grade, norm_len, norm_poly])
            except Exception as e:
                logger.debug(f"Feature extraction failed for text '{text[:20]}...': {e}")
                features.append([0.0, 0.0, 0.0])  # Fallback

        return torch.tensor(features, dtype=torch.float32)
