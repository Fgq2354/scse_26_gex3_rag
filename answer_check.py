"""Evidence checks shared by the agent and offline tests."""

UNKNOWN = "I don't know."


def problems(answer, passages):
    text = answer["answer"].strip()
    quote = answer["quote"].strip()
    policy_id = answer["policy_id"].strip()
    if not text:
        return ["The answer is empty"]
    if answer["unknown"]:
        errors = []
        if text != UNKNOWN:
            errors.append("An unknown answer must contain only I don't know.")
        if quote or policy_id:
            errors.append("An unknown answer must not attach an unrelated citation")
        return errors
    matched = next((p for p in passages if p["policy_id"] == policy_id), None)
    if matched is None:
        return ["The cited policy was not retrieved"]
    if not quote or quote not in matched["text"]:
        return ["The evidence quote does not occur verbatim in the retrieved policy"]
    if text != quote:
        return ["An extractive answer must match its evidence quote exactly"]
    if text == UNKNOWN:
        return ["The unknown flag contradicts the answer"]
    return []
