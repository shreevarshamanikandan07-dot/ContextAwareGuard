"""
Explanation Generator: Human-Readable Evidence-Grounded Explanations.
Synthesizes outputs from G1, G2, G3, and the DistilBERT baseline classifier
to generate actionable, transparent natural language explanations for operators.
"""


class ExplanationGenerator:
    def generate_explanation(self, analysis_result):
        """
        Generates a structured, evidence-grounded explanation based on the pipeline output.
        """
        is_injection = analysis_result.get("is_injection", False)
        risk_score = analysis_result.get("risk_score", 0)
        risk_tier = analysis_result.get("risk_tier", "LOW")
        declared_task = analysis_result.get("declared_task", "unknown").upper()
        
        g1 = analysis_result.get("g1_context", {})
        g2 = analysis_result.get("g2_risk", {})
        g3 = analysis_result.get("g3_spans", {})

        evidence_bullets = []
        action_recommendation = ""

        # 1. Base Model & Overall Verdict
        if is_injection:
            headline = f"PROMPT INJECTION DETECTED ({risk_tier} RISK: {risk_score}/100)"
            action_recommendation = "BLOCK or REDACT: The prompt contains adversarial instructions attempting to hijack the host LLM."
        elif risk_tier == "MEDIUM":
            headline = f"AMBIGUOUS / SUSPICIOUS INSTRUCTION ({risk_tier} RISK: {risk_score}/100)"
            action_recommendation = "SANDBOX / AUDIT: The prompt exhibits instruction-like syntax but lacks clear override intent."
        else:
            headline = f"BENIGN INPUT VERIFIED ({risk_tier} RISK: {risk_score}/100)"
            action_recommendation = "ALLOW: The prompt is consistent with legitimate user data for the declared task."

        # 2. G1 Evidence: Task Semantic Alignment & Same-Task Evasion
        host_sim = g1.get("host_semantic_similarity", 0.0)
        is_same_task = g1.get("is_same_task_evasion_candidate", False)
        drift_score = g1.get("instruction_drift_score", 0.0)

        if is_same_task:
            evidence_bullets.append(
                f"[G1 Same-Task Analysis]: Prompt exhibits high domain overlap with host task '{declared_task}' "
                f"(similarity: {host_sim*100:.1f}%), but contains an embedded instruction drift score of {drift_score:.2f}, "
                f"indicating a covert same-domain injection attempt."
            )
        elif is_injection:
            evidence_bullets.append(
                f"[G1 Context Analysis]: Input introduces external directive instructions diverging from declared task '{declared_task}'."
            )
        else:
            evidence_bullets.append(
                f"[G1 Context Analysis]: Input text aligns normally with expected data inputs for task '{declared_task}'."
            )

        # 3. G2 Evidence: Structural Context & Quotation Safety
        q_safety = g2.get("quotation_safety", {})
        rep_speech = g2.get("reported_speech", {})
        delim_escape = g2.get("delimiter_escape_detected", False)

        if q_safety.get("is_safely_quoted", False):
            evidence_bullets.append(
                f"[G2 Benign Mitigation]: Detected imperative keywords are safely enclosed within quotation marks "
                f"as passive data payload (Quoted ratio: {q_safety.get('quote_ratio', 0)*100:.1f}%). False positive suppressed."
            )
        elif rep_speech.get("is_reported_speech", False) and not is_injection:
            evidence_bullets.append(
                f"[G2 Benign Mitigation]: Sentence utilizes 3rd-person reported/narrative structure typical of benign premises. Risk calibrated down."
            )
        elif delim_escape:
            evidence_bullets.append(
                f"[G2 Adversarial Escalation]: Detected explicit markdown, code block, or system prompt delimiter breakout patterns."
            )
        else:
            boundary_state = "unquoted direct instruction payload" if is_injection else "standard unescaped text input"
            evidence_bullets.append(
                f"[G2 Boundary Analysis]: Text evaluated as {boundary_state}; calibrated risk score: {risk_score}/100."
            )

        # 4. G3 Evidence: Span Localization
        spans = g3.get("spans", [])
        if spans:
            span_summaries = [f"'{s['span_text']}' ({s['reason']})" for s in spans[:2]]
            evidence_bullets.append(
                f"[G3 Span Localization]: {len(spans)} suspicious span(s) pinpointed: {'; '.join(span_summaries)}."
            )
        else:
            evidence_bullets.append(
                f"[G3 Span Localization]: No adversarial instruction spans identified."
            )

        summary_paragraph = (
            f"The input was analyzed against host task '{declared_task}'. "
            f"The integrated ContextAwarePromptGuard assigned a calibrated risk score of {risk_score}/100 ({risk_tier} tier). "
            f"{action_recommendation}"
        )

        return {
            "headline": headline,
            "summary": summary_paragraph,
            "evidence": evidence_bullets,
            "recommendation": action_recommendation,
            "risk_tier": risk_tier,
            "risk_score": risk_score
        }
