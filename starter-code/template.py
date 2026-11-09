"""
Lab #4: System Prompt Engineering & Tool Calling Engine.
Mock agent for product search and customer support in the Vingroup ecosystem.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_MAP

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT cấp sản xuất: persona, công cụ, quy tắc và phạm vi hoạt động.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant, trợ lý tư vấn thân thiện và chính xác về các sản phẩm,
dịch vụ trong hệ sinh thái Vingroup.

## PERSONA
- Tư vấn bằng tiếng Việt, chuyên nghiệp, dễ hiểu và lịch sự.
- Phân biệt rõ thông tin đã được xác nhận với thông tin chưa có dữ liệu.

## AVAILABLE TOOLS
- search_product_catalog: Tra cứu sản phẩm/dịch vụ theo danh mục và giá tối đa.
- submit_support_ticket: Ghi nhận yêu cầu hỗ trợ của khách hàng.

## CORE RULES
1. Không bịa tên sản phẩm, giá, chính sách, tình trạng xử lý hoặc mã ticket.
2. Khi cần tìm sản phẩm hay giá, phải dùng search_product_catalog.
3. Chỉ xác nhận ticket đã tạo sau khi submit_support_ticket trả về thành công.
4. Nếu công cụ không có dữ liệu hoặc trả về lỗi, hãy nói rõ điều đó.
5. Chỉ khẳng định thông tin sản phẩm dựa trên Observation hoặc dữ liệu FAQ
   được cung cấp rõ ràng.

## OPERATIONAL BOUNDARIES
- Chỉ hỗ trợ câu hỏi liên quan đến Vingroup, bao gồm VinFast và Vinpearl.
- Với yêu cầu ngoài phạm vi, lịch sự nêu giới hạn và mời người dùng hỏi về
  sản phẩm/dịch vụ Vingroup.

## OUTPUT CONTRACT
- Theo dõi từng bước theo dạng Thought → Action → Observation.
- Trả lời cuối cùng bằng tiếng Việt, ngắn gọn và bám sát các Observation.
- Không hiển thị suy luận nội bộ; chỉ cung cấp kết luận và dữ kiện cần thiết.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # Baseline mock không gọi tool, dùng để đối chiếu với agent có tool.
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        normalized_input = user_input.lower()
        needs_catalog = (
            any(term in normalized_input for term in ("xe điện", "xe vinfast", "vinfast", "du lịch", "vinpearl", "resort"))
            and any(term in normalized_input for term in ("xem", "tìm", "giá", "có ", "dưới", "tối đa", "đặt"))
            and "bảo hành" not in normalized_input
        )
        needs_ticket = any(
            term in normalized_input
            for term in (
                "tạo ticket", "cần hỗ trợ", "xin hỗ trợ", "khiếu nại",
                "bị lỗi", "lỗi hệ thống", "hỏng", "không hoạt động",
                "sự cố", "trục trặc"
            )
        )

        self.trace.append({
            "step": "init",
            "user_input": user_input,
            "intents": {
                "needs_catalog": needs_catalog,
                "needs_ticket": needs_ticket
            }
        })

        actions = []
        if needs_catalog:
            category = (
                "du_lich"
                if any(term in normalized_input for term in ("du lịch", "vinpearl", "resort"))
                else "xe_dien"
            )
            max_price = 999999999999
            price_match = re.search(
                r"(?:dưới|tối đa|không quá|thấp hơn|giá)\s*"
                r"([\d.,]+)\s*(tỷ|tỉ|triệu|tr)?",
                normalized_input
            )
            if price_match:
                raw_amount = price_match.group(1)
                if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", raw_amount):
                    amount = float(raw_amount.replace(",", "").replace(".", ""))
                else:
                    amount = float(raw_amount.replace(",", "."))
                unit = price_match.group(2)
                if unit in ("triệu", "tr"):
                    amount *= 1_000_000
                elif unit in ("tỷ", "tỉ"):
                    amount *= 1_000_000_000
                max_price = int(amount)
            actions.append((
                "search_product_catalog",
                {"category": category, "max_price": max_price}
            ))

        if needs_ticket:
            name_match = re.search(
                r"(?:tôi tên|tên tôi là|tôi là|em tên|mình tên)\s+"
                r"([A-ZÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*){1,4})",
                user_input,
                re.IGNORECASE
            )
            customer_name = name_match.group(1).strip(" ,.!?") if name_match else "Khách hàng"
            issue_description = user_input
            if name_match:
                issue_description = (
                    user_input[:name_match.start()]
                    + user_input[name_match.end():]
                ).strip(" ,.!?")
            priority = (
                "high"
                if any(term in normalized_input for term in ("nghiêm trọng", "gấp", "khẩn cấp", "khẩn"))
                else "low"
                if any(term in normalized_input for term in ("không gấp", "ít khẩn", "mức thấp"))
                else "medium"
            )
            actions.append((
                "submit_support_ticket",
                {
                    "customer_name": customer_name,
                    "issue_description": issue_description or user_input,
                    "priority": priority
                }
            ))

        if not actions:
            answer = self._faq_answer(user_input)
            if self.max_iterations < 1:
                answer = "Lỗi: Vượt quá số bước tối đa."
                self.trace.append({"step": "max_iterations_reached", "answer": answer})
                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": 0,
                    "status": "max_iterations_reached"
                }
            self.trace.append({"step": "final", "iteration": 1, "answer": answer})
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        iteration = 1
        action_index = 0
        observations = {}
        while iteration <= self.max_iterations:
            if action_index < len(actions):
                tool_name, arguments = actions[action_index]
                observation = TOOL_MAP[tool_name](**arguments)
                observations[tool_name] = observation
                self.trace.append({
                    "step": "tool_call",
                    "iteration": iteration,
                    "tool": tool_name,
                    "arguments": arguments,
                    "observation": observation
                })
                action_index += 1

                if action_index == len(actions) and len(actions) == 1:
                    answer = self._format_tool_answer(tool_name, observation)
                    self.trace.append({
                        "step": "final",
                        "iteration": iteration,
                        "answer": answer
                    })
                    return {
                        "answer": answer,
                        "trace": self.trace,
                        "iterations": iteration,
                        "status": "completed"
                    }
                iteration += 1
                continue

            answer = self._format_combined_answer(observations)
            self.trace.append({
                "step": "final",
                "iteration": iteration,
                "answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        answer = "Lỗi: Vượt quá số bước tối đa."
        self.trace.append({"step": "max_iterations_reached", "answer": answer})
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": self.max_iterations,
            "status": "max_iterations_reached"
        }

    @staticmethod
    def _format_tool_answer(tool_name: str, observation: Any) -> str:
        if tool_name == "search_product_catalog":
            if isinstance(observation, list) and observation and "error" in observation[0]:
                return f"Không thể tra cứu sản phẩm: {observation[0]['error']}"
            if not observation:
                return "Rất tiếc, không tìm thấy sản phẩm phù hợp."
            products = [
                f"- {product['name']}: {product['price_vnd']:,} VNĐ"
                for product in observation
            ]
            return "Các sản phẩm phù hợp:\n" + "\n".join(products)

        return (
            f"Đã tạo yêu cầu hỗ trợ cho {observation['customer_name']}. "
            f"Mã ticket: {observation['ticket_id']} "
            f"(ưu tiên {observation['priority']})."
        )

    @classmethod
    def _format_combined_answer(cls, observations: Dict[str, Any]) -> str:
        parts = [
            cls._format_tool_answer(tool_name, observation)
            for tool_name, observation in observations.items()
        ]
        return "\n\n".join(parts)

    @staticmethod
    def _faq_answer(user_input: str) -> str:
        normalized_input = user_input.lower()
        if "bảo hành" in normalized_input and any(
            term in normalized_input for term in ("pin", "xe điện", "vinfast")
        ):
            return (
                "Thông tin trong catalog hiện có ghi pin VinFast VF 5 Plus "
                "được bảo hành 10 năm. Chính sách cụ thể có thể tùy mẫu xe "
                "và điều kiện áp dụng; vui lòng xác nhận với VinFast."
            )
        if any(term in normalized_input for term in ("vingroup", "vinfast", "vinpearl")):
            return (
                "Mình chưa có đủ dữ liệu để xác nhận thông tin này. "
                "Bạn có thể hỏi về sản phẩm trong catalog hoặc gửi yêu cầu hỗ trợ."
            )
        return (
            "Mình chỉ hỗ trợ thông tin về các sản phẩm và dịch vụ thuộc "
            "hệ sinh thái Vingroup."
        )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
