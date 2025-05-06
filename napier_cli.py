#!/usr/bin/env python3
import asyncio
import sys
import os
import json
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv
import google.generativeai as genai
from rich.console import Console
from rich.markdown import Markdown
from rich import print as rprint
from rich.panel import Panel
from rich.prompt import Prompt

# Load environment variables from .env file
load_dotenv()

# Initialize Rich console for better terminal output
console = Console()

class NapierClient:
    """
    Napier - An MCP client that connects AI models with third-party applications.
    Currently supports Gemini model integration.
    """
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Initialize Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            console.print("[bold red]Error: GEMINI_API_KEY not found in environment variables.[/bold red]")
            console.print("[yellow]Please create a .env file with your GEMINI_API_KEY=[/yellow]")
            sys.exit(1)
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Conversation history
        self.chat_history = []
        self.connected_server = None
    
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        # Store the server path
        self.connected_server = server_script_path
        
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,          # Python or Node interpreter
            args=[server_script_path], # Path to server script
            env=None                  # Use current environment
        )

        try:
            console.print(f"[yellow]Connecting to MCP server: {server_script_path}...[/yellow]")
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

            # Initialize the session
            await self.session.initialize()

            # List available tools
            response = await self.session.list_tools()
            tools = response.tools
            tool_names = [tool.name for tool in tools]
            
            console.print(f"[green]Successfully connected to server![/green]")
            console.print(Panel(
                f"[bold]Available tools:[/bold]\n" + "\n".join([f"• {name}" for name in tool_names]),
                title="MCP Server Connection",
                border_style="green"
            ))
            
            return tool_names
        except Exception as e:
            console.print(f"[bold red]Error connecting to server: {str(e)}[/bold red]")
            return None
    
    async def process_query(self, query: str) -> str:
        """Process a query using Gemini and available tools"""
        if not self.session:
            return "Error: Not connected to any MCP server. Use 'connect' command first."
        
        # Add user query to history
        self.chat_history.append({"role": "user", "parts": [query]})
        
        # Get available tools from the server
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
        
        # Format tools for Gemini
        tools_description = "You have access to the following tools:\n\n"
        for tool in available_tools:
            tools_description += f"- {tool['name']}: {tool['description']}\n"
            tools_description += f"  Input schema: {json.dumps(tool['input_schema'], indent=2)}\n\n"
        
        # Prepare prompt with instructions and tools
        system_prompt = f"""You are an AI assistant that helps users interact with various applications through tools.
{tools_description}

INSTRUCTIONS:
1. Analyze the user's request carefully.
2. If a tool is needed to fulfill the request, decide which tool to use.
3. Format your tool calls as JSON, wrapped in triple backticks with the 'json' tag.
4. Example tool call format:
```json
{{
  "tool_name": "tool_name_here",
  "parameters": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}
```
5. After receiving tool results, provide a helpful response that incorporates the information.
6. If no tool is needed, respond directly to the user's request.

Always make sure to follow the exact input schema for each tool when making a call."""

        try:
            # Initialize Gemini chat
            chat = self.model.start_chat(history=self.chat_history)
            
            # Send the query along with system prompt
            response = chat.send_message(
                [system_prompt, query],
                generation_config={"temperature": 0.2}
            )
            
            response_text = response.text
            self.chat_history.append({"role": "model", "parts": [response_text]})
            
            # Process tool calls in response
            import re
            tool_call_pattern = r"```json\s*(\{[^`]*\})\s*```"
            tool_calls = re.findall(tool_call_pattern, response_text)
            
            final_response = []
            
            if tool_calls:
                for tool_call_json in tool_calls:
                    try:
                        tool_call = json.loads(tool_call_json)
                        tool_name = tool_call.get("tool_name")
                        parameters = tool_call.get("parameters", {})
                        
                        console.print(f"[bold cyan]Executing tool:[/bold cyan] {tool_name}")
                        console.print(f"[cyan]Parameters:[/cyan] {json.dumps(parameters, indent=2)}")
                        
                        # Execute tool call
                        result = await self.session.call_tool(tool_name, parameters)
                        
                        # Format tool result for display
                        result_str = f"\n[Tool Result: {tool_name}]\n{result.content}\n"
                        final_response.append(result_str)
                        
                        # Send tool result back to Gemini
                        followup_system_prompt = f"""The tool '{tool_name}' returned the following result:

{result.content}

Please analyze this result and provide a helpful response to the user based on this information."""

                        followup_response = chat.send_message(followup_system_prompt)
                        final_response.append(followup_response.text)
                        self.chat_history.append({"role": "model", "parts": [followup_response.text]})
                        
                    except json.JSONDecodeError:
                        console.print(f"[bold red]Error: Invalid JSON format in tool call[/bold red]")
                        final_response.append("Error: Invalid tool call format detected.")
                    except Exception as e:
                        console.print(f"[bold red]Error executing tool: {str(e)}[/bold red]")
                        final_response.append(f"Error executing tool: {str(e)}")
                
                return "\n".join(final_response)
            else:
                # No tool calls, just return the response
                return response_text
                
        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
            return f"Error processing query: {str(e)}"

    async def chat_loop(self):
        """Run an interactive chat loop"""
        welcome_message = """
        █▄░█ ▄▀█ █▀█ █ █▀▀ █▀█
        █░▀█ █▀█ █▀▀ █ ██▄ █▀▄
        
        Model Context Protocol Client
        
        Type:
        • 'connect <path_to_server>' to connect to an MCP server
        • 'exit' or 'quit' to exit
        • Any other text to send as a query to the AI
        """
        console.print(Panel(welcome_message, border_style="blue"))

        while True:
            try:
                if self.connected_server:
                    prompt = f"[bold blue]Napier[/bold blue] ({os.path.basename(self.connected_server)}) > "
                else:
                    prompt = "[bold blue]Napier[/bold blue] > "
                    
                user_input = Prompt.ask(prompt)
                
                if user_input.lower() in ['exit', 'quit']:
                    console.print("[yellow]Exiting Napier...[/yellow]")
                    break
                
                elif user_input.lower().startswith('connect '):
                    server_path = user_input[8:].strip()
                    await self.connect_to_server(server_path)
                    
                elif user_input.lower() == 'help':
                    help_text = """
                    Available commands:
                    • 'connect <path_to_server>' - Connect to an MCP server
                    • 'exit' or 'quit' - Exit the application
                    • 'help' - Display this help message
                    • Any other text will be processed as a query to the AI
                    """
                    console.print(Panel(help_text, title="Napier Help", border_style="green"))
                    
                elif user_input.strip():
                    if not self.session:
                        console.print("[bold yellow]Not connected to any MCP server. Use 'connect <path_to_server>' first.[/bold yellow]")
                        continue
                        
                    response = await self.process_query(user_input)
                    console.print(Panel(Markdown(response), title="AI Response", border_style="cyan"))
                    
            except Exception as e:
                console.print(f"[bold red]Error: {str(e)}[/bold red]")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
        console.print("[green]Resources cleaned up.[/green]")

async def main():
    """Main entry point"""
    # Display ASCII art banner
    banner = """
    ███╗   ██╗ █████╗ ██████╗ ██╗███████╗██████╗ 
    ████╗  ██║██╔══██╗██╔══██╗██║██╔════╝██╔══██╗
    ██╔██╗ ██║███████║██████╔╝██║█████╗  ██████╔╝
    ██║╚██╗██║██╔══██║██╔═══╝ ██║██╔══╝  ██╔══██╗
    ██║ ╚████║██║  ██║██║     ██║███████╗██║  ██║
    ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝
    """
    console.print(Panel(banner, border_style="blue"))
    console.print("[bold]Napier[/bold] - The MCP Client connecting AI with applications")
    console.print("Version 1.0.0\n")

    client = NapierClient()
    try:
        # If a server path is provided as an argument, connect to it
        if len(sys.argv) > 1:
            await client.connect_to_server(sys.argv[1])
        
        # Start the chat loop
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Keyboard interrupt detected. Exiting Napier...[/yellow]")