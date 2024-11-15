import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from datetime import date
import yfinance as yf
import pandas as pd
from langchain_core.tools import tool, StructuredTool
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Get Groq API key from environment
groq_api_key = os.environ.get("GROQ_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

import math

def sanitize_ratios(ratios):
    for key, value in ratios.items():
        if isinstance(value, float) and math.isnan(value):
            ratios[key] = None  # Convert NaN to null for JSON compatibility
    return ratios


# Financial ratio calculation functions
def calculate_ratios_for_year(balance_sheet, income_statement, cash_flow, year, annual_dividends, ticker):
    ratios = {}

    def get_value(data, row, year):
        if row in data.index and year in data.columns:
            value = data.loc[row, year]
            return value.iloc[0] if not value.empty else None
        return None

    # Extract financial data
    current_assets = get_value(balance_sheet, 'Current Assets', year)
    current_liabilities = get_value(balance_sheet, 'Current Liabilities', year)
    total_assets = get_value(balance_sheet, 'Total Assets', year)
    total_liabilities = get_value(balance_sheet, 'Total Liabilities Net Minority Interest', year)
    total_equity = get_value(balance_sheet, 'Stockholders Equity', year)
    inventory = get_value(balance_sheet, 'Inventory', year)
    receivables = get_value(balance_sheet, 'Accounts Receivable', year)
    ebit = get_value(income_statement, 'EBIT', year)
    interest_expense = get_value(income_statement, 'Interest Expense', year)
    net_income = get_value(income_statement, 'Net Income', year)
    revenue = get_value(income_statement, 'Total Revenue', year)
    operating_cash_flow = get_value(cash_flow, 'Operating Cash Flow', year)
    cash = get_value(balance_sheet, 'Cash And Cash Equivalents', year)
    cost_of_revenue = get_value(income_statement, 'Reconciled Cost Of Revenue', year)

    # Additional data for dividends and market price
    shares_outstanding = ticker.info.get("sharesOutstanding")
    market_price = ticker.info.get("regularMarketPrice")

    # Calculate ratios
    if current_assets is not None and current_liabilities is not None:
        ratios['Current Ratio'] = round(current_assets / current_liabilities, 2)
    if current_assets is not None and current_liabilities is not None and inventory is not None:
        ratios['Acid-Test Ratio'] = round((current_assets - inventory) / current_liabilities, 2)
    if cash is not None and current_liabilities is not None:
        ratios['Cash Ratio'] = round(cash / current_liabilities, 2)
        ratios['Current Cash Ratio'] = round(cash / current_liabilities, 2)
    if total_assets is not None and total_liabilities is not None:
        ratios['Debt Ratio'] = round(total_liabilities / total_assets, 2)
    if total_liabilities is not None and total_equity is not None:
        ratios['Debt to Equity Ratio'] = round(total_liabilities / total_equity, 2)
    if ebit is not None and interest_expense is not None:
        ratios['Interest Coverage Ratio'] = round(ebit / interest_expense, 2)
    if operating_cash_flow is not None and total_liabilities is not None:
        ratios['Debt Service Coverage Ratio'] = round(operating_cash_flow / total_liabilities, 2)
    if revenue is not None and total_assets is not None:
        ratios['Asset Turnover Ratio'] = round(revenue / total_assets, 2)
    if revenue is not None and inventory is not None:
        ratios['Inventory Turnover Ratio'] = round(revenue / inventory, 2)
    if revenue is not None and receivables is not None:
        ratios['Receivables Turnover Ratio'] = round(revenue / receivables, 2)
    if ratios.get('Inventory Turnover Ratio') is not None:
        ratios['Days Sales in Inventory'] = round(365 / ratios['Inventory Turnover Ratio'], 2)
    if revenue is not None:
        if cost_of_revenue is not None:
            ratios['Gross Margin Ratio'] = round((revenue - cost_of_revenue) / revenue, 2)
        if ebit is not None:
            ratios['Operating Margin Ratio'] = round(ebit / revenue, 2)
    if net_income is not None and total_assets is not None:
        ratios['Return on Assets (ROA)'] = round(net_income / total_assets, 2)
    if net_income is not None and total_equity is not None:
        ratios['Return on Equity (ROE)'] = round(net_income / total_equity, 2)
    if total_equity is not None and shares_outstanding is not None:
        ratios['Book Value per Share'] = round(total_equity / shares_outstanding, 2)
    if annual_dividends is not None and shares_outstanding is not None and market_price is not None:
        dividends_per_share = annual_dividends / shares_outstanding
        ratios['Dividend Yield'] = round(dividends_per_share / market_price, 2)
    if net_income is not None and shares_outstanding is not None:
        ratios['Earnings per Share (EPS)'] = round(net_income / shares_outstanding, 2)
    if ratios.get('Earnings per Share (EPS)') is not None and market_price is not None:
        ratios['Price-Earnings Ratio (P/E)'] = round(market_price / ratios['Earnings per Share (EPS)'], 2)
    if operating_cash_flow is not None and total_liabilities is not None:
        ratios['Operating Cash Flow Ratio'] = round(operating_cash_flow / total_liabilities, 2)

    return sanitize_ratios(ratios)

def calculate_ratios(ticker_data):
    balance_sheet = ticker_data.balance_sheet
    income_statement = ticker_data.financials
    cash_flow = ticker_data.cashflow
    ticker=ticker_data

    fiscal_years = ['2019', '2020', '2021', '2022', '2023', '2024']
    all_ratios = {}

    dividends = ticker_data.dividends.resample('Y').sum()
    
    for year in fiscal_years:
        if year in balance_sheet.columns and year in income_statement.columns and year in cash_flow.columns:
            year_dt = pd.to_datetime(f"{year}-12-31")
            annual_dividends = dividends.get(year_dt, None)
            all_ratios[year] = calculate_ratios_for_year(balance_sheet, income_statement, cash_flow, year, annual_dividends, ticker)
        else:
            print(f"Data not available for {year}")
            all_ratios[year] = "Data not available for this year"
    return sanitize_ratios(all_ratios)

@tool
def financial_ratios(ticker: str) -> dict:
    """Retrieve information on financial ratios for a company."""
    ticker_obj = yf.Ticker(ticker)
    return calculate_ratios(ticker_obj)

# Define additional financial tools using yfinance
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
        financial_ratios,
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
            ("system",
             "You are a helpful assistant. Try to answer user queries using the available tools. If a tool is not available for a specific query, tell the user that the tool is not available but provide your best answer."),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    llama3 = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192", temperature=0)
    return create_tool_calling_agent(llama3, tools, prompt)

def handle_tool_error(exception: Exception) -> str:
    # Generate a response using the LLM when the tool fails
    return "Tool is not available, but here is what I think: " + str(exception)

agent_executor = AgentExecutor(
    agent=initialize_agent(),
    tools=[financial_ratios,
        company_information,
        last_dividend_and_earnings_date,
        stock_splits_history,
        summary_of_mutual_fund_holders,
        summary_of_institutional_holders,
        stock_grade_updrages_downgrades,
        stock_news,],
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

    if "everything" in user_prompt:
        # Extract ticker symbol from the prompt (simple example)
        company_name = user_prompt.replace("everything", "").strip()
        if company_name:
            ratios = financial_ratios(company_name.upper())
            #ratios = financial_ratios('TATACOMM.NS')
            temp=jsonify({'response': ratios})
            return temp
    
    try:
        response = agent_executor.invoke({"messages": [HumanMessage(content=user_prompt)]})
        answer = response["output"]
        return jsonify({'response': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
