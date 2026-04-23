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
        """
        
        stream=True 后，response 不是完整答案，而是一个数据流（生成器）
        AI 每生成一个字 / 一个词，就会传过来一小块数据（chunk）
        我们用 for 循环挨个接收这些小块数据
        逐行详解
        1. for chunk in response:
        作用：遍历流式响应的每一个数据片段
        chunk = 每次接收到的一小段 AI 输出（可能是一个字、一个标点）
        循环会一直执行，直到 AI 把完整回答生成完毕
        2. content=chunk["message"]["content"]
        每个 chunk 都是一个字典，格式固定：
        {"message": {"content": "等", ...}}
        这行代码的作用：从数据块中，提取出 AI 刚生成的文字
        比如第一次拿到 等，第二次拿到 于，第三次拿到 2
        3. ai_reply+=content
        ai_reply 是我们提前定义的空字符串
        +=：把每次提取的文字拼接累加
        目的：收集完整的 AI 回答，后面要存入对话历史，让 AI 记住

        
        """
        for chunk in response:
            content=chunk["message"]["content"]
            ai_reply+=content
            print(content,end="",flush=True)
        print("\n")

        messages.append({"role":"assistant","content":ai_reply})

if __name__ == "__main__":
    ai_assistant_with_role()
