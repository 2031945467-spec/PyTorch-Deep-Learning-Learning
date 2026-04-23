import ollama

def ai_assistant_with_role():
    print("=" * 60)
    print("🤖 专业AI助手（角色已设定 | 流式输出 | 输入q退出/clear清空记忆）")
    print("=" * 60)

    messages=[
        {
            "role":"system",
            "content":"你是一个简洁专业的AI助手，回答简短精准、不啰嗦，语气友好，只讲重点"
        }
    ]

    while True:
        user_input=input("你：").strip()

        if user_input=="q":
            print("✅ 助手下线！")
            break
        if user_input=="clear":
            messages=[messages[0]]
            print("🧹 对话记忆已清空，角色设定保留\n")
            continue

        messages.append({"role":"user","content":user_input})

        print("AI: ",end="",flush=True)
        ai_reply=""
        response=ollama.chat(
            model="qwen2:0.5b",
            messages=messages,
            stream=True
        )
        for chunk in response:
            content=chunk["message"]["content"]
            ai_reply+=content
            print(content,end="",flush=True)
        print("\n")

        messages.append({"role":"assistant","content":ai_reply})

if __name__ == "__main__":
    ai_assistant_with_role()