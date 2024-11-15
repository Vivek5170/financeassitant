import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from datetime import date
from langchain_core.tools import tool, StructuredTool
# Load environment variables
load_dotenv()

# Get Groq API key from environment
groq_api_key = os.environ.get("GROQ_API_KEY")

# Import necessary modules for the Finance Agent
import yfinance as yf
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Define financial tools using yfinance
@tool
def company_information(ticker: str) -> dict:
    """Use this tool to retrieve company information like address, industry, sector, company officers, business summary, website,
       marketCap, current price, ebitda, total debt, total revenue, debt-to-equity, etc."""

    ticker_obj = yf.Ticker(ticker)
    ticker_info = ticker_obj.get_info()

    return ticker_info

@tool
def last_dividend_and_earnings_date(ticker: str) -> dict:
    """
    Use this tool to retrieve company's last dividend date and earnings release dates.
    It does not provide information about historical dividend yields.
    """
    ticker_obj = yf.Ticker(ticker)

    return ticker_obj.get_calendar()

@tool
def summary_of_mutual_fund_holders(ticker: str) -> dict:
    """
    Use this tool to retrieve company's top mutual fund holders.
    It also returns their percentage of share, stock count and value of holdings.
    """
    ticker_obj = yf.Ticker(ticker)
    mf_holders = ticker_obj.get_mutualfund_holders()

    return mf_holders.to_dict(orient="records")

@tool
def summary_of_institutional_holders(ticker: str) -> dict:
    """
    Use this tool to retrieve company's top institutional holders.
    It also returns their percentage of share, stock count and value of holdings.
    """
    ticker_obj = yf.Ticker(ticker)
    inst_holders = ticker_obj.get_institutional_holders()

    return inst_holders.to_dict(orient="records")

@tool
def stock_grade_updrages_downgrades(ticker: str) -> dict:
    """
    Use this to retrieve grade ratings upgrades and downgrades details of particular stock.
    It'll provide name of firms along with 'To Grade' and 'From Grade' details. Grade date is also provided.
    """
    ticker_obj = yf.Ticker(ticker)

    curr_year = date.today().year

    upgrades_downgrades = ticker_obj.get_upgrades_downgrades()
    upgrades_downgrades = upgrades_downgrades.loc[upgrades_downgrades.index > f"{curr_year}-01-01"]
    upgrades_downgrades = upgrades_downgrades[upgrades_downgrades["Action"].isin(["up", "down"])]

    return upgrades_downgrades.to_dict(orient="records")

@tool
def stock_splits_history(ticker: str) -> dict:
    """
    Use this tool to retrieve company's historical stock splits data.
    """
    ticker_obj = yf.Ticker(ticker)
    hist_splits = ticker_obj.get_splits()

    return hist_splits.to_dict()

@tool
def stock_news(ticker: str) -> dict:
    """
    Use this to retrieve latest news articles discussing particular stock ticker.
    """
    ticker_obj = yf.Ticker(ticker)

    return ticker_obj.get_news()

# Initialize the Finance Agent
def initialize_agent():
    tools = [
        company_information,
        last_dividend_and_earnings_date,
        stock_splits_history,
        summary_of_mutual_fund_holders,
        summary_of_institutional_holders,
        stock_grade_updrages_downgrades,
        stock_news,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Try to answer user queries using the available tools. If a tool is not available for a specific query, tell the user that the tool is not available but provide your best answer.",
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Initialize Llama-3 LLM via Groq API
    llama3 = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192", temperature=0)

    finance_agent = create_tool_calling_agent(llama3, tools, prompt)
    return finance_agent

# Custom error handling function
def handle_tool_error(exception: Exception) -> str:
    # Generate a response using the LLM when the tool fails
    return "Tool is not available, but here is what I think: " + str(exception)

# Instantiate the agent executor with custom error handling
agent_executor = AgentExecutor(
    agent=initialize_agent(),
    tools=[
        company_information,
        last_dividend_and_earnings_date,
        stock_splits_history,
        summary_of_mutual_fund_holders,
        summary_of_institutional_holders,
        stock_grade_updrages_downgrades,
        stock_news,
    ],
    handle_tool_error=handle_tool_error,
    verbose=False,
)

# Define the API endpoint
@app.route('/prompt', methods=['POST'])
def handle_prompt():
    data = request.get_json()
    user_prompt = data.get('prompt', '')

    if not user_prompt:
        return jsonify({'error': 'Prompt is required.'}), 400

    # Process the prompt using the Finance Agent
    try:
        response = agent_executor.invoke({"messages": [HumanMessage(content=user_prompt)]})
        answer = response["output"]
        return jsonify({'response': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
