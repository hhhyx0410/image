from flask import Flask, request, render_template, session, jsonify
from flask_session import Session
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import torch
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import initialize_agent



# 确保环境变量正确加载
load_dotenv()

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


# 初始化 OpenAI API 的基础和密钥
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name='gpt-3.5-turbo')

# 图像描述模型初始化
device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_to_text_model = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(image_to_text_model)
model = BlipForConditionalGeneration.from_pretrained(image_to_text_model).to(device)


def describeImage(image_url):
    try:
        image_object = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
        inputs = processor(image_object, return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        return processor.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"An error occurred: {str(e)}"


class DescribeImageTool(BaseTool):
    name = "Describe Image Tool"
    description = "Use this tool to describe an image."

    def _run(self, url: str):
        return describeImage(url)

    def _arun(self, query: str):
        raise NotImplementedError("Async operation not supported yet")


tools = [DescribeImageTool()]


agent = initialize_agent(
    llm=llm,
    tools=tools,
    config={
        'verbose': True,
        'max_iterations': 3,
        'early_stopping_method': 'generate',
        'memory_config': {
            'memory_key': 'chat_history',
            'k': 5,
            'return_messages': True
        },
        'handle_parsing_errors': True  # Ensure parsing errors are handled internally
    }
)



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        image_url = request.form['image_url']
        description = describeImage(image_url)
        session['image_url'] = image_url
        session['description'] = description
        return render_template('index.html', image_url=image_url, description=description)
    return render_template('index.html')


@app.route('/chat', methods=['POST', 'GET'])
def chat():
    try:
        if request.method == 'POST':
            user_input = request.form['chat_input']
            response = agent.invoke(user_input)  # 使用新的invoke方法，并处理可能的解析错误

            # 假设 response 结构中包含 'final_answer' 作为最终答案
            final_answer = response.get('final_answer', "No answer provided")
            session['final_answer'] = final_answer  # 存储最终答案到 session

            return render_template('index.html', final_answer=final_answer, image_url=session.get('image_url'), description=session.get('description'))
        return render_template('index.html', final_answer=session.get('final_answer', None), image_url=session.get('image_url'), description=session.get('description'))
    except Exception as e:
        app.logger.error(f"An error occurred in chat: {str(e)}")
        return str(e), 500




if __name__ == '__main__':
    app.run(debug=True)

