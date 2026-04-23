import requests
import json

def cloud_llm_chat():
    print("=" * 60)
    print("☁️ 云端大模型聊天机器人 | q 退出 | clear 清空记忆")
    print("=" * 60)

    messages = [
        {
            "role":"system",
            "content":"你是一个专业简洁的AI助手，回答准确、简短、不啰嗦。"
        }
    ]

    api_url = "https://api.siliconflow.cn/v1/chat/completions"
    model = "Qwen/Qwen2-1.5B-Instruct"

    while True:
        user_input=input("你: ").strip()
        if user_input=="q":
            print("✅ 对话结束")
            break
        if user_input=="clear":
            messages=[messages[0]]
            print("🧹 已清空对话历史\n")
            continue
        messages.append({"role":"user","content":user_input})

        headers={"Content-Type":"application/json"}

        payload={
            "model":model,
            "messages":messages,
            "stream":True,
            "temperature":0.7
        }

        print("AI: ",end="",flush=True)
        ai_reply=""

        try:
            response=requests.post(url=api_url,json=payload,headers=headers,stream=True)
            for line in response.iter_lines():
                if not line:
                    continue
                line=line.decode("utf-8")
                if line.startswith("data: "):
                    data_str=line.replace("data: ","")
                    if data_str=="[DONE]":
                        break
                    try:
                        data=json.loads(data_str)
                        delta=data["choices"][0]["delta"]
                        content=delta.get("content","")
                        ai_reply+=content
                        print(content,end="",flush=True)
                    except:
                        continue
        except Exception as e:
            print(f"\n❌ 调用失败，请检查网络：{e}")
            print("💡 提示：云端API可能需要密钥，我们下节课换回本地强化版")

if __name__ == "__main__":
    cloud_llm_chat()