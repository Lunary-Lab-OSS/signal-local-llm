"""
Intent Router Component
Cascading router architecture with three layers:
1. Reflex: Pattern matching (0ms latency)
2. Estimator: Difficulty score calculation (~8-10ms latency - GTE Base EN v1.5 + matrix regressor)
3. Thinker: Dynamic thinking budget based on difficulty (quadratic curve)
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class IntentRouter:
    """
    Cascading router architecture with three layers:
    1. Reflex: Pattern matching (0ms latency) - bypasses all neural computation
    2. Estimator: Difficulty score calculation (4ms latency) - Arctic-XS embedding + matrix regressor
    3. Thinker: Dynamic thinking budget (quadratic curve) - maps difficulty to token budget
    """

    def __init__(
        self,
        router_type: str = "llm",
        routellm_controller: Any | None = None,
        routellm_threshold: float = 0.5,
        routellm_router_name: str = "mf",
        min_thinking_tokens: int = 0,
        max_thinking_tokens: int = 2500,
        coreml_classifier: Any | None = None,
    ):
        """
        Initialize IntentRouter.

        Args:
            router_type: "routellm", "coreml", or "llm" - MUST match config
            routellm_controller: RouteLLM controller (REQUIRED if router_type="routellm")
            routellm_threshold: difficulty threshold (0.0-1.0). Reused as the
                complexity threshold when router_type="coreml".
            routellm_router_name: RouteLLM router name (e.g., "mf") - used to access router from controller
            coreml_classifier: CoreMLComplexityClassifier (REQUIRED if router_type="coreml")
        """
        self._reflex_patterns = self._build_reflex_patterns()
        self._semantic_keywords = self._build_semantic_keywords()
        self._router_model = None
        self._router_type = router_type
        self._coreml_classifier = coreml_classifier

        # Thinking budget configuration (quadratic curve mapping)
        self._min_thinking_tokens = min_thinking_tokens
        self._max_thinking_tokens = max_thinking_tokens

        # Validate RouteLLM configuration
        if router_type == "routellm":
            if routellm_controller is None:
                raise RuntimeError(
                    "FATAL: RouteLLM is configured but controller is None. RouteLLM controller is REQUIRED."
                )
            self._routellm_controller = routellm_controller
            self._routellm_threshold = routellm_threshold
            self._routellm_router_name = routellm_router_name
            logger.info(
                f"✅ RouteLLM router initialized (router={routellm_router_name}, threshold={routellm_threshold})"
            )
            logger.info(
                f"   Thinking budget: {min_thinking_tokens}-{max_thinking_tokens} tokens (quadratic curve)"
            )
        elif router_type == "coreml":
            if coreml_classifier is None:
                raise RuntimeError(
                    "FATAL: router_type='coreml' but coreml_classifier is None. The Core ML complexity classifier is REQUIRED."
                )
            self._routellm_threshold = routellm_threshold  # reused as complexity threshold
            logger.info(
                f"✅ Core ML complexity router initialized (threshold={routellm_threshold})"
            )
            logger.info(
                f"   Thinking budget: {min_thinking_tokens}-{max_thinking_tokens} tokens (quadratic curve)"
            )
        elif router_type == "llm":
            logger.info("✅ LLM-based router configured")
        else:
            raise RuntimeError(
                f"FATAL: Invalid router_type '{router_type}'. Must be 'routellm', 'coreml', or 'llm'."
            )

    def _build_reflex_patterns(self) -> set[str]:
        """Build optimized reflex command patterns"""
        return {
            # Window management
            "close window",
            "open window",
            "minimize",
            "maximize",
            "restore",
            "switch window",
            "next window",
            "previous window",
            # Navigation
            "scroll up",
            "scroll down",
            "scroll left",
            "scroll right",
            "page up",
            "page down",
            "go to top",
            "go to bottom",
            # Actions
            "click",
            "double click",
            "right click",
            "type",
            "paste",
            "copy",
            "cut",
            "select all",
            "undo",
            "redo",
            "save",
            "open",
            "close",
            # System
            "lock screen",
            "sleep",
            "shutdown",
            "restart",
            # Simple greetings only - questions go to router model
            "hello",
            "hi",
            "hey",
            "thanks",
            "thank you",
            "yes",
            "no",
            "ok",
            "okay",
        }

    def _build_semantic_keywords(self) -> set[str]:
        """Build semantic command keywords"""
        # NOTE: Removed overly generic words like "for", "if", "while", "return", "else"
        # that appear in normal English text and cause false positives
        # These should only match in code-specific contexts, not general questions
        return {
            # Code structure (removed generic words that appear in normal English)
            "function",
            "class",
            "method",
            "variable",
            "parameter",
            "import",
            "try",
            "except",
            # Code actions
            "delete",
            "rename",
            "move",
            "extract",
            "refactor",
            "inline",
            "find",
            "replace",
            "go to",
            "jump to",
            "navigate",
            # Code context
            "definition",
            "declaration",
            "usage",
            "reference",
        }

    def route(
        self,
        text: str,
        router_model: Any | None = None,
        router_generate_func: Callable[[str], str] | None = None,
    ) -> tuple[str, float | None, int | None]:
        """
        Cascading router: Layer 1 (Reflex) -> Layer 2 (Estimator) -> Layer 3 (Thinker)

        Args:
            text: Input text to classify
            router_model: LLM model (REQUIRED if router_type="llm", ignored if router_type="routellm")
            router_generate_func: LLM generation function (REQUIRED if router_type="llm", ignored if router_type="routellm")

        Returns:
            Tuple of (intent, difficulty_score, thinking_tokens):
            - intent: "reflex", "semantic", or "agentic"
            - difficulty_score: Float 0.0-1.0 (None for reflex/semantic)
            - thinking_tokens: Integer token budget (None for reflex/semantic)
        """
        text_lower = text.lower().strip()

        # LAYER 1: Reflex (0ms latency) - bypasses all neural computation
        if self._is_reflex(text_lower):
            logger.debug(f"Route: REFLEX (0ms) - '{text_lower}'")
            return ("reflex", None, None)

        # LAYER 2: Estimator (4ms latency) - calculate difficulty score
        # Only if not reflex, calculate difficulty using RouteLLM router
        difficulty_score = None
        if self._router_type == "routellm":
            # Use RouteLLM router to get difficulty score (GTE Base EN v1.5 embedding + matrix regressor)
            router = self._routellm_controller.routers[self._routellm_router_name]
            difficulty_score = router.calculate_strong_win_rate(text)
            logger.debug(
                "Route: Difficulty score = %.3f (~8-10ms GTE Base EN v1.5 + matrix)",
                difficulty_score,
            )
        elif self._router_type == "coreml":
            # Use the Core ML DeBERTa-v3 classifier's prompt_complexity_score (0-1)
            # as the difficulty signal (~25-30ms on CPU for the 128-token variant).
            difficulty_score = self._coreml_classifier.complexity_score(text)
            logger.debug("Route: Complexity score = %.3f (Core ML CPU)", difficulty_score)
        elif self._router_type == "llm":
            # For LLM-based routing, we can't get a continuous score easily
            # Default to 0.5 (medium difficulty) for now
            difficulty_score = 0.5
            logger.debug("Route: Using default difficulty score 0.5 (LLM router)")

        # Check semantic keywords (fast, no embedding needed)
        if self._is_semantic(text_lower):
            logger.debug(f"Route: SEMANTIC (keyword match) - '{text_lower}'")
            return ("semantic", None, None)

        # LAYER 3: Thinker - map difficulty to thinking budget
        # Try to use MF model's thinking regressor if available, otherwise fallback to difficulty-based estimation
        thinking_tokens = None
        if difficulty_score is not None:
            # Check if router has thinking regressor (only if using RouteLLM)
            if (
                self._router_type == "routellm"
                and hasattr(self, "_routellm_controller")
                and self._routellm_controller is not None
                and hasattr(self._routellm_controller, "routers")
                and self._routellm_router_name in self._routellm_controller.routers
            ):
                router = self._routellm_controller.routers[self._routellm_router_name]
                if hasattr(router, "model") and hasattr(router.model, "predict_thinking_tokens"):
                    try:
                        # Use MF model's thinking regressor if available
                        thinking_tokens = router.model.predict_thinking_tokens(
                            text,
                            difficulty_score=difficulty_score,
                            min_tokens=self._min_thinking_tokens,
                            max_tokens=self._max_thinking_tokens,
                        )
                        logger.debug(
                            f"Route: Using MF model thinking prediction: {thinking_tokens} tokens"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Route: MF thinking prediction failed ({e}), falling back to difficulty-based estimation"
                        )
                        thinking_tokens = self._calculate_thinking_budget(difficulty_score)
                else:
                    # No thinking regressor - use fallback
                    thinking_tokens = self._calculate_thinking_budget(difficulty_score)
            else:
                # No router available - use fallback
                thinking_tokens = self._calculate_thinking_budget(difficulty_score)

        # Determine intent based on difficulty score (only for RouteLLM or CoreML)
        if difficulty_score is not None and self._router_type in ("routellm", "coreml"):
            if difficulty_score > self._routellm_threshold:
                intent = "agentic"
                logger.info(
                    f"Route: AGENTIC (difficulty={difficulty_score:.3f}, thinking_tokens={thinking_tokens})"
                )
            else:
                # Low difficulty -> semantic (no thinking)
                intent = "semantic"
                logger.info(
                    f"Route: SEMANTIC (difficulty={difficulty_score:.3f} <= threshold {self._routellm_threshold:.3f})"
                )
        else:
            # Fallback to LLM-based classification if RouteLLM not available
            if self._router_type == "llm":
                if router_generate_func:
                    intent = self._classify_with_llm_func(text_lower, router_generate_func)
                elif router_model:
                    intent = self._classify_with_llm(text_lower, router_model)
                else:
                    raise RuntimeError(
                        "FATAL: LLM router is configured but router_model/router_generate_func is None."
                    )
            else:
                intent = "agentic"  # Default to agentic if we can't determine

        return (intent, difficulty_score, thinking_tokens)

    def _calculate_thinking_budget(self, difficulty_score: float) -> int:
        """
        Map difficulty score (0.0-1.0) to thinking token budget using quadratic curve.

        Formula: T_budget = T_min + (T_max - T_min) * (Score^2)

        This ensures:
        - Low scores (0.1) stay low (~25 tokens)
        - Medium scores (0.5) get moderate budget (~625 tokens)
        - High scores (0.9) get maximum budget (~2025 tokens)

        Args:
            difficulty_score: Float between 0.0 (Easy) and 1.0 (Impossible)

        Returns:
            Integer token budget
        """
        # Clamp score to [0.0, 1.0]
        score = max(0.0, min(1.0, difficulty_score))

        # Quadratic curve: score^2 makes low scores stay low, high scores ramp up fast
        score_squared = score * score
        budget = (
            self._min_thinking_tokens
            + (self._max_thinking_tokens - self._min_thinking_tokens) * score_squared
        )

        return int(budget)

    def _is_reflex(self, text_lower: str) -> bool:
        """Fast pattern matching for reflex commands - STRICT matching only"""
        # Only check exact matches - no substring matching to avoid false positives
        # Questions and complex queries should go to router model for proper classification
        # For multi-word commands, only match if it's an exact phrase match
        # Don't do substring matching - let router decide for ambiguous cases
        # This prevents "How fast is this?" from matching "is" or "this"
        return text_lower in self._reflex_patterns

    def _is_semantic(self, text_lower: str) -> bool:
        """Keyword detection for semantic commands - requires multiple keywords or code-specific phrases"""
        # Check for semantic keywords
        words = set(text_lower.split())
        matching_keywords = words & self._semantic_keywords

        # Require at least 2 keywords to avoid false positives from common words
        # OR match code-specific phrases like "go to function", "find definition"
        if len(matching_keywords) >= 2:
            return True

        # Check for code-specific phrases (more reliable than single words)
        code_phrases = [
            "go to",
            "jump to",
            "find definition",
            "find declaration",
            "go to function",
            "go to class",
            "go to method",
            "find usage",
            "find reference",
        ]
        return any(phrase in text_lower for phrase in code_phrases)

    def _classify_with_llm(self, text: str, router_model: Any) -> str:
        """Classify using LLM router - router must decide, no fallback"""
        # Router model is required - no fallbacks
        if not hasattr(router_model, "generate"):
            raise RuntimeError(
                "Router model doesn't support generate() method. Cannot classify intent."
            )

        # Improved prompt for better classification
        prompt = f"Classify this command into one category:\nCommand: {text}\n\nCategories:\n- reflex: Simple commands like 'click', 'scroll', 'close window'\n- semantic: Code navigation like 'go to function', 'find definition'\n- agentic: Complex queries requiring reasoning like 'how do I...', 'explain...', 'write code for...'\n\nAnswer with only one word:"

        try:
            # Generate classification
            if "ExLlamaV2DynamicGenerator" in str(type(router_model)):
                result = router_model.generate(prompt, max_new_tokens=5)
            else:
                # Fallback for other models (Transformers, MLX)
                result = router_model.generate(prompt, max_tokens=5, temperature=0.1)

            result_lower = result.strip().lower()

            # Router must explicitly choose - no default fallback
            if "reflex" in result_lower:
                return "reflex"
            elif "semantic" in result_lower:
                return "semantic"
            elif "agentic" in result_lower:
                return "agentic"
            else:
                # Router didn't give a clear answer - this is an error
                raise RuntimeError(
                    f"Router model returned unclear classification: '{result}'. Expected: reflex, semantic, or agentic"
                )
        except RuntimeError:
            raise  # Re-raise RuntimeErrors
        except Exception as e:
            raise RuntimeError(
                f"Router model classification failed: {e}. Router model is required and must work correctly."
            ) from e

    def _classify_with_routellm(self, text: str) -> str:
        """Classify using RouteLLM for fast routing decision (<50ms) - NO FALLBACKS"""
        # RouteLLM decides: strong model (agentic) vs weak model (reflex/semantic)
        # Access the router directly from the controller using router name
        router = self._routellm_controller.routers[self._routellm_router_name]

        # Calculate strong win rate using the router's method
        strong_win_rate = router.calculate_strong_win_rate(text)

        # Log RouteLLM decision for debugging
        logger.info(
            f"🔀 RouteLLM win_rate: {strong_win_rate:.3f} (threshold: {self._routellm_threshold:.3f})"
        )

        if strong_win_rate > self._routellm_threshold:
            # RouteLLM says: needs strong model -> agentic
            logger.info(
                f"   → Routing to AGENTIC (win_rate {strong_win_rate:.3f} > threshold {self._routellm_threshold:.3f})"
            )
            return "agentic"
        else:
            # RouteLLM says: weak model is enough -> check if reflex or semantic
            # Use pattern matching to distinguish reflex vs semantic
            if self._is_semantic(text):
                logger.info(
                    f"   → Routing to SEMANTIC (win_rate {strong_win_rate:.3f} <= threshold)"
                )
                return "semantic"
            else:
                # Default to reflex for simple commands
                logger.info(f"   → Routing to REFLEX (win_rate {strong_win_rate:.3f} <= threshold)")
                return "reflex"

    def _classify_with_llm_func(self, text: str, router_generate_func: Callable[[str], str]) -> str:
        """Classify using router generation function (proper settings applied)"""
        # Improved prompt for better classification
        prompt = f"Classify this command into one category:\nCommand: {text}\n\nCategories:\n- reflex: Simple commands like 'click', 'scroll', 'close window'\n- semantic: Code navigation like 'go to function', 'find definition'\n- agentic: Complex queries requiring reasoning like 'how do I...', 'explain...', 'write code for...'\n\nAnswer with only one word:"

        try:
            result = router_generate_func(prompt)
            result_lower = result.strip().lower()

            # Router must explicitly choose - no default fallback
            if "reflex" in result_lower:
                return "reflex"
            elif "semantic" in result_lower:
                return "semantic"
            elif "agentic" in result_lower:
                return "agentic"
            else:
                # Router didn't give a clear answer - this is an error
                raise RuntimeError(
                    f"Router model returned unclear classification: '{result}'. Expected: reflex, semantic, or agentic"
                )
        except RuntimeError:
            raise  # Re-raise RuntimeErrors
        except Exception as e:
            raise RuntimeError(
                f"Router model classification failed: {e}. Router model is required and must work correctly."
            ) from e
