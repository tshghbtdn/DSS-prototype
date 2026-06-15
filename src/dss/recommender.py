class ModelRecommender:

    def __init__(self, rules=None):

        self.rules = rules or []

    def recommend(
        self,
        demand_type: str,
        cv: float,
        mean_sales: float
    ):

        return {
            "model": None,
            "confidence": None,
            "evidence": "Recommendation rules chưa được cập nhật.",
            "reasoning": (
                "Đang chờ cập nhật."
                ""
            )
        }