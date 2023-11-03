# Clinical Exam AI Analysis System

## Overview

This AI analysis system is designed to assist medical students in understanding and learning the diagnostic process. It analyzes clinical exam questions and provides explanations for the diagnoses, helping students to improve their diagnostic reasoning and explanation skills.

## Features

- **Clinical Case Analysis**: Takes a clinical case description and a list of possible diseases to identify key reasons supporting the diagnosis.
- **AI-Driven Explanations**: Utilizes AI to generate explanations, aiding in the understanding of why a particular diagnosis is correct.
- **Interactive UI**: A user-friendly web interface allows for easy input of clinical cases and displays the AI-generated explanations.
- **Educational Tool**: Serves as a learning aid for medical students to enhance their clinical reasoning skills.

## How It Works

1. **Input**: Users enter a clinical case along with a list of potential diagnoses.
2. **Processing**: The AI system analyzes the input and uses NLP techniques to extract key information.
3. **Explanation Generation**: The system generates explanations for the correct diagnosis and provides reasoning for discarding other possibilities.
4. **Output**: The results are presented in an interactive format, highlighting the named entities and offering detailed explanations.

## Technology Stack

- **Frontend**: HTML, Bootstrap, JavaScript
- **AI Pipeline**: Huggingface

## Installation and Usage

To get the Clinical Exam AI Analysis System running on your local machine, follow the steps outlined below:

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment tool (optional, but recommended)

### Setting Up the Environment
1. Clone the repository or download the source code to your local machine.
2. Navigate to the project directory.

    ```bash
    cd path/to/project
    ```

3. (Optional) Set up a virtual environment to isolate the project dependencies.

    ```bash
    # For Windows
    python -m venv venv
    # For macOS/Linux
    python3 -m venv venv
    ```

4. Activate the virtual environment.

    ```bash
    # For Windows
    venv\Scripts\activate
    # For macOS/Linux
    source venv/bin/activate
    ```

### Install Dependencies
Install the required libraries by executing:

```bash
pip install -r requirements.txt
```

### Set Up the `OPENAI_API_KEY` Environment Variable
The system requires an OpenAI API key to function correctly. Set the `OPENAI_API_KEY` environment variable using the command line:

```bash
# For Windows
set OPENAI_API_KEY=your_openai_api_key_here

# For macOS/Linux
export OPENAI_API_KEY=your_openai_api_key_here
```

Replace `your_openai_api_key_here` with your actual API key provided by OpenAI.

### Launch the System
Once the environment is set up and the dependencies are installed, start the demo system with:

```bash
flask run
```

This command will launch a local server. Access the system by navigating to `http://127.0.0.1:5000/` in your web browser.

### Deactivating the Virtual Environment
When you're done working with the system, you can deactivate the virtual environment by simply running:

```bash
deactivate
```

## Interface

To add images

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

The MIT License is a permissive free software license originating at the Massachusetts Institute of Technology (MIT). It allows you to:

- **Use**: You are free to use this software for both commercial and private projects.
- **Copy**: You can make as many copies of the source code as you wish.
- **Modify**: You have the liberty to modify the original source code to fit your needs.
- **Distribute**: You can incorporate all or part of the software in your own programs. You can also offer the software as-is or with modifications, provided you include the MIT license terms with it.

## Contact

Santiago Marro - smarro@gmail.com
---
