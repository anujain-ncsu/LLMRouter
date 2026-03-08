"""
Prompt Analyzer — heuristic-based prompt classification
for automatic model selection.
"""

import re

from middleware.models import ModelRecommendation


# ── Keyword sets for classification ────────────────────────────────────────

CODE_KEYWORDS = {
    "def ", "function ", "class ", "import ", "require(", "from ",
    "const ", "let ", "var ", "async ", "await ", "return ",
    "if __name__", "print(", "console.log", "System.out",
    "```", "public static", "void main", "#include",
    "SELECT ", "INSERT ", "UPDATE ", "DELETE ", "CREATE TABLE",
    "interface ", "implements ", "extends ", "template",
}

CODE_PATTERNS = [
    r"```[\w]*\n",              # Code blocks
    r"def\s+\w+\s*\(",          # Python function defs
    r"function\s+\w+\s*\(",     # JS function defs
    r"class\s+\w+",             # Class defs
    r"\w+\.\w+\(.*\)",          # Method calls
    r"for\s*\(.+\)",            # For loops
    r"while\s*\(.+\)",          # While loops
    r"import\s+\w+",            # Imports
    r"\{[\s\S]*:\s*[\s\S]*\}",  # Object literals
]

REASONING_KEYWORDS = {
    "analyze", "analyse", "compare", "contrast", "explain why",
    "evaluate", "assess", "critique", "argue", "debate",
    "pros and cons", "advantages", "disadvantages", "trade-off",
    "implications", "consequences", "hypothesis", "theory",
    "because", "therefore", "however", "furthermore",
    "in conclusion", "reasoning", "logic", "evidence",
    "step by step", "break down", "elaborate",
}

CASUAL_KEYWORDS = {
    "hi", "hello", "hey", "what's up", "how are you",
    "thanks", "thank you", "bye", "goodbye", "see ya",
    "lol", "haha", "nice", "cool", "awesome",
    "tell me a joke", "fun fact", "random",
}

CASUAL_PATTERNS = [
    r"^(hi|hello|hey|yo|sup)\b",
    r"^(thanks|thank you|thx)\b",
    r"^what'?s up\b",
]


def _score_code(prompt: str) -> float:
    """Score how likely the prompt is requesting code-related work."""
    prompt_lower = prompt.lower()
    score = 0.0

    # Keyword matches
    for keyword in CODE_KEYWORDS:
        if keyword.lower() in prompt_lower:
            score += 0.15

    # Pattern matches
    for pattern in CODE_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            score += 0.2

    # Explicit code requests
    code_requests = ["write code", "write a function", "implement",
                     "debug", "fix this code", "refactor", "code review",
                     "write a script", "programming", "algorithm"]
    for req in code_requests:
        if req in prompt_lower:
            score += 0.3

    return min(score, 1.0)


def _score_reasoning(prompt: str) -> float:
    """Score how likely the prompt requires deep reasoning/analysis."""
    prompt_lower = prompt.lower()
    score = 0.0

    # Keyword matches
    for keyword in REASONING_KEYWORDS:
        if keyword in prompt_lower:
            score += 0.12

    # Long prompts are more likely analytical
    word_count = len(prompt.split())
    if word_count > 50:
        score += 0.15
    elif word_count > 100:
        score += 0.25

    # Question complexity
    question_words = ["why", "how does", "what causes", "what is the relationship",
                      "explain the difference", "what are the implications"]
    for qw in question_words:
        if qw in prompt_lower:
            score += 0.15

    return min(score, 1.0)


def _score_casual(prompt: str) -> float:
    """Score how likely the prompt is casual/chat."""
    prompt_lower = prompt.lower().strip()
    score = 0.0

    # Keyword matches
    for keyword in CASUAL_KEYWORDS:
        if keyword in prompt_lower:
            score += 0.2

    # Pattern matches
    for pattern in CASUAL_PATTERNS:
        if re.search(pattern, prompt_lower):
            score += 0.3

    # Short prompts are more likely casual
    word_count = len(prompt.split())
    if word_count <= 5:
        score += 0.3
    elif word_count <= 10:
        score += 0.15

    return min(score, 1.0)


def analyze_prompt(prompt: str) -> ModelRecommendation:
    """
    Analyze a prompt and recommend the best model for it.

    Uses heuristic scoring across code, reasoning, and casual categories.
    Falls back to the general-purpose model if no category scores high.

    Returns:
        ModelRecommendation with model_id, confidence, and reasoning.
    """
    code_score = _score_code(prompt)
    reasoning_score = _score_reasoning(prompt)
    casual_score = _score_casual(prompt)

    scores = {
        "stellarcode-70b": (code_score, "code/technical"),
        "quantumleap-13b": (reasoning_score, "reasoning/analysis"),
        "nebulachat-3b": (casual_score, "casual/chat"),
    }

    # Find top-scoring category
    best_model = max(scores, key=lambda k: scores[k][0])
    best_score, best_category = scores[best_model]

    # Minimum confidence threshold: if nothing scores well, default to general
    CONFIDENCE_THRESHOLD = 0.25

    if best_score < CONFIDENCE_THRESHOLD:
        return ModelRecommendation(
            model_id="novamind-7b",
            confidence=0.5,
            reasoning="No strong signal detected; using general-purpose model.",
        )

    return ModelRecommendation(
        model_id=best_model,
        confidence=round(best_score, 3),
        reasoning=f"Prompt classified as {best_category} (score: {best_score:.2f}).",
    )
