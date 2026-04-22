import ollama

def llm_chat_with_memory():
    print("=" * 60)
    print("🔥 LLM 多轮对话助手（支持上下文记忆，输入 clear 清空记忆，q 退出）")
    print("=" * 60)

    messages=[]

    while True:
        user_input=input("你: ").strip()
        if user_input.lower()=="q":
            print("✅ 对话结束，AI 下线！")
            break

        if user_input.lower()=="clear":
            messages=[]
            print("✅ 对话结束，AI 下线！")
            continue

        messages.append({"role":"user","content":user_input})

        response=ollama.chat(
            model="qwen2:0.5b",
            messages=messages
        )

        ai_reply=response["message"]["content"]
        print(f"AI：{ai_reply}\n")

        messages.append({"role":"assistant","content":ai_reply})

if __name__ == "__main__":
    llm_chat_with_memory()
