# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This is a Gradio app that includes two tabs: one for code generation and another for resource management.
# The resource management tab has been updated to allow file uploads, deletion, and a table listing all the files.
# Additionally, three small text boxes have been added for managing file dataframe parameters.

import argparse
import os
from pathlib import Path
import gradio as gr
import requests
import pandas as pd
import os
import uvicorn
import json
import argparse
# from utils import build_logger, make_temp_image, server_error_msg, split_video
from urllib.parse import urlparse
from pathlib import Path
from fastapi import FastAPI
# from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# logger = build_logger("gradio_web_server", "gradio_web_server.log")
logflag = os.getenv("LOGFLAG", False)

# create a FastAPI app
app = FastAPI()
cur_dir = os.getcwd()
static_dir = Path(os.path.join(cur_dir, "static/"))
tmp_dir = Path(os.path.join(cur_dir, "split_tmp_videos/"))

Path(static_dir).mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

tmp_upload_folder = "/tmp/gradio/"



host_ip = os.getenv("host_ip")
DATAPREP_REDIS_PORT = os.getenv("DATAPREP_REDIS_PORT", 6007)
DATAPREP_ENDPOINT = os.getenv("DATAPREP_ENDPOINT", f"http://{host_ip}:{DATAPREP_REDIS_PORT}/v1/dataprep")
MEGA_SERVICE_PORT = os.getenv("MEGA_SERVICE_PORT", 7778)

backend_service_endpoint = os.getenv(
        "BACKEND_SERVICE_ENDPOINT", f"http://{host_ip}:{MEGA_SERVICE_PORT}/v1/codegen"
    )

dataprep_ingest_endpoint = f"{DATAPREP_ENDPOINT}/ingest"
dataprep_get_files_endpoint = f"{DATAPREP_ENDPOINT}/get"
dataprep_delete_files_endpoint = f"{DATAPREP_ENDPOINT}/delete"
dataprep_get_indices_endpoint = f"{DATAPREP_ENDPOINT}/indices"



# Define the functions that will be used in the app
def conversation_history(prompt, index, use_agent, history):
    # Print the language and prompt, and return a placeholder code
    print(f"Generating code for prompt: {prompt} using index: {index} and use_agent is {use_agent}")
    history.append([prompt, ""])
    response_generator = generate_code(prompt, index, use_agent)
    for token in response_generator:
        history[-1][-1] += token
        yield history


def upload_media(media, index=None, chunk_size=1500, chunk_overlap=100):
    media = media.strip().split("\n")
    print("Files passed is ", media, flush=True)
    if not chunk_size:
        chunk_size = 1500
    if not chunk_overlap:
        chunk_overlap = 100

    requests = []
    if type(media) is list:
        for file in media:
            file_ext = os.path.splitext(file)[-1]
            if is_valid_url(file):
                print(file, " is valid URL")
                print("Ingesting URL...")
                yield (
                    gr.Textbox(
                        visible=True,
                        value="Ingesting URL...",
                    )
                )
                value = ingest_url(file, index, chunk_size, chunk_overlap)
                requests.append(value)
                yield value
            elif file_ext in ['.pdf', '.txt']:
                print("Ingesting File...")
                yield (
                    gr.Textbox(
                        visible=True,
                        value="Ingesting file...",
                    )
                )
                value = ingest_file(file, index, chunk_size, chunk_overlap)
                requests.append(value)
                yield value
            else:
                print(file, "File type not supported")
                yield (
                    gr.Textbox(
                        visible=True,
                        value="Your media is either an invalid URL or the file extension type is not supported. (Supports .pdf, .txt, url)",
                    )
                )
                return
        yield requests

    else:
        file_ext = os.path.splitext(media)[-1]
        if is_valid_url(media):
            value = ingest_url(media, index, chunk_size, chunk_overlap)
            yield value
        elif file_ext in ['.pdf', '.txt']:
            print("Ingesting File...")
            value = ingest_file(media, index, chunk_size, chunk_overlap)
            # print("Return value is: ", value, flush=True)
            yield value
        else:
            print(media, "File type not supported")
            yield (
                gr.Textbox(
                    visible=True,
                    value="Your file extension type is not supported.",
                )
            )
            return

def generate_code(query, index=None, use_agent=False):
    if index is None or index == "None":
        input_dict = {"messages": query, "agents_flag": use_agent}
    else:
        input_dict = {"messages": query, "index_name": index, "agents_flag": use_agent}

    print("Query is ", input_dict)
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url=backend_service_endpoint, headers=headers, data=json.dumps(input_dict), stream=True)

    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith("data: "):  # Only process lines starting with "data: "
                json_part = line[len("data: "):]  # Remove the "data: " prefix
                if json_part.strip() == "[DONE]":  # Ignore the DONE marker
                    continue
                try:
                    json_obj = json.loads(json_part)  # Convert to dictionary
                    if "choices" in json_obj:
                        for choice in json_obj["choices"]:
                            if "text" in choice:
                                # Yield each token individually
                                yield choice["text"]
                except json.JSONDecodeError:
                    print("Error parsing JSON:", json_part)


def ingest_file(file, index=None, chunk_size=100, chunk_overlap=150):
    headers = {
         # "Content-Type: multipart/form-data"
        }
    file_input = {"files": open(file, "rb")}

    if index:
        print("Index is", index)
        data = {"index_name": index, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    else:
        data = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}

    print("Calling Request Now!")
    response = requests.post(url=dataprep_ingest_endpoint, headers=headers, files=file_input, data=data)
    # print("Ingest Files", response)
    print(response.text)
        
    # table = update_table()
    return response.text

def ingest_url(url, index=None, chunk_size=100, chunk_overlap=150):
    print("URL is ", url)
    url = str(url)
    if not is_valid_url(url):
        return "Invalid URL entered. Please enter a valid URL"
    
    headers = {
         # "Content-Type: multipart/form-data"
        }

    if index:
        url_input = {"link_list": json.dumps([url]), "index_name": index, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    else:
        url_input = {"link_list": json.dumps([url]), "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    response = requests.post(url=dataprep_ingest_endpoint, headers=headers, data=url_input)
    # print("Ingest URL", response)
    # table = update_table()
    return response.text


def is_valid_url(url):
    url = str(url)
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False



# Initialize the file list
file_list = []

# def update_files(file):
#     # Add the uploaded file to the file list
#     file_list.append(file.name)
#     file_df["Files"] = file_list
#     return file_df


def get_files(index=None):
    headers = {
        # "Content-Type: multipart/form-data"
    }
    if index == "All Files":
        index = None

    if index:
        index = {"index_name": index}
        response = requests.post(url=dataprep_get_files_endpoint, headers=headers, data=index)
        print("Get files with ", index, response)
        table = response.json()
        return table
    else:
        # print("URL IS ", dataprep_get_files_endpoint)
        response = requests.post(url=dataprep_get_files_endpoint, headers=headers)
        print("Get files ", response)
        table = response.json()
        return table

def update_table(index=None):
    if index == "All Files":
        index = None
    files = get_files(index)
    print("Files is ", files)
    if len(files) == 0:
        df = pd.DataFrame(files, columns=["Files"])
        return df
    else:
        df = pd.DataFrame(files)
        return df
    
def update_indices():
    indices = get_indices()
    df = pd.DataFrame(indices, columns=["File Indices"])
    return df

def delete_file(file, index=None):
    # Remove the selected file from the file list
    headers = {
        # "Content-Type: application/json"
    }
    print("URL IS ", dataprep_delete_files_endpoint)
    if index:
        file_input = {"files": open(file, "rb"), "index_name": index}
    else:
        file_input = {"files": open(file, "rb")}
    response = requests.post(url=dataprep_delete_files_endpoint, headers=headers, data=file_input)
    print("Delete file ", response)
    table = update_table()
    return response.text

def delete_all_files(index=None):
    # Remove all files from the file list
    headers = {
        # "Content-Type: application/json"
    }
    response = requests.post(url=dataprep_delete_files_endpoint, headers=headers, data='{"file_path": "all"}')
    print("Delete all files ", response)
    table = update_table()
    
    return "Delete All status: " + response.text

def get_indices():
    headers = {
        # "Content-Type: application/json"
    }
    print("URL IS ", dataprep_get_indices_endpoint)
    response = requests.post(url=dataprep_get_indices_endpoint, headers=headers)
    indices = ["None"]
    print("Get Indices", response)
    indices += response.json()
    return indices

def update_indices_dropdown():
    new_dd = gr.update(choices=get_indices(), value="None")
    return new_dd
    

def get_file_names(files):
    file_str = ""
    if not files:
        return file_str
    
    for file in files:
      file_str += file + '\n'
    file_str.strip()
    return file_str


# Define UI components
with gr.Blocks() as ui:
    with gr.Tab("Code Generation"):
        gr.Markdown("### Generate Code from Natural Language")
        chatbot = gr.Chatbot(label="Chat History")
        prompt_input = gr.Textbox(label="Enter your query")
        with gr.Column():
            with gr.Row(equal_height=True):
                # indices = ["None"] + get_indices()
                database_dropdown = gr.Dropdown(choices=get_indices(), label="Select Index", value="None", scale=10)
                db_refresh_button = gr.Button("Refresh Dropdown", scale=0.1)
                db_refresh_button.click(update_indices_dropdown, outputs=database_dropdown)
                use_agent = gr.Checkbox(label="Use Agent", container=False)
            # with gr.Row(scale=1):
            
        
        generate_button = gr.Button("Generate Code")

        # Connect the generate button to the conversation_history function
        generate_button.click(conversation_history, inputs=[prompt_input, database_dropdown, use_agent, chatbot], outputs=chatbot)

    with gr.Tab("Resource Management"):
        # File management components
        # url_button = gr.Button("Process")
        with gr.Row():
            with gr.Column(scale=1):
                index_name_input = gr.Textbox(label="Index Name")
                chunk_size_input = gr.Textbox(label="Chunk Size", value="1500", placeholder="Enter an integer (default: 1500)")
                chunk_overlap_input = gr.Textbox(label="Chunk Overlap", value="100", placeholder="Enter an integer (default: 100)")
            with gr.Column(scale=3):
                file_upload = gr.File(label="Upload Files", file_count="multiple")
                url_input = gr.Textbox(label="Media to be ingested (Append URL's in a new line)")
                upload_button = gr.Button("Upload", variant="primary")
                upload_status = gr.Textbox(label="Upload Status")
                file_upload.change(get_file_names, inputs=file_upload, outputs=url_input)
            with gr.Column(scale=1):
                # table_dropdown = gr.Dropdown(indices)
                # file_table = gr.Dataframe(interactive=False, value=update_table())
                file_table = gr.Dataframe(interactive=False, value=update_indices())
                refresh_button = gr.Button("Refresh", variant="primary", size="sm")
                refresh_button.click(update_indices, outputs=file_table)
                # refresh_button.click(update_indices, outputs=database_dropdown)
                # table_dropdown.change(fn=update_table, inputs=table_dropdown, outputs=file_table)
                # upload_button.click(upload_media, inputs=[file_upload, index_name_input, chunk_size_input, chunk_overlap_input], outputs=file_table)
                upload_button.click(upload_media, inputs=[url_input, index_name_input, chunk_size_input, chunk_overlap_input], outputs=upload_status)
                
                delete_all_button = gr.Button("Delete All", variant="primary", size="sm")
                delete_all_button.click(delete_all_files, outputs=upload_status)
        
        
        
                # delete_button = gr.Button("Delete Index")

                # selected_file_output = gr.Textbox(label="Selected File")
                # delete_button.click(delete_file, inputs=indices, outputs=upload_status)

      

ui.queue()
app = gr.mount_gradio_app(app, ui, path="/")
share = False
enable_queue = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=os.getenv("UI_PORT", 5173))
    parser.add_argument("--concurrency-count", type=int, default=20)
    parser.add_argument("--share", action="store_true")

    host_ip = os.getenv("host_ip")
    DATAPREP_REDIS_PORT = os.getenv("DATAPREP_REDIS_PORT", 6007)
    DATAPREP_ENDPOINT = os.getenv("DATAPREP_ENDPOINT", f"http://{host_ip}:{DATAPREP_REDIS_PORT}/v1/dataprep")
    MEGA_SERVICE_PORT = os.getenv("MEGA_SERVICE_PORT", 7778)


    backend_service_endpoint = os.getenv(
        "BACKEND_SERVICE_ENDPOINT", f"http://{host_ip}:{MEGA_SERVICE_PORT}/v1/codegen"
    )

    # dataprep_ingest_endpoint = f"{DATAPREP_ENDPOINT}/ingest"
    # dataprep_get_files_endpoint = f"{DATAPREP_ENDPOINT}/get"
    # dataprep_delete_files_endpoint = f"{DATAPREP_ENDPOINT}/delete"
    # dataprep_get_indices_endpoint = f"{DATAPREP_ENDPOINT}/indices"


    args = parser.parse_args()
    # logger.info(f"args: {args}")
    global gateway_addr
    gateway_addr = backend_service_endpoint
    global dataprep_ingest_addr
    dataprep_ingest_addr = dataprep_ingest_endpoint
    global dataprep_get_files_addr
    dataprep_get_files_addr = dataprep_get_files_endpoint


    uvicorn.run(app, host=args.host, port=args.port)
