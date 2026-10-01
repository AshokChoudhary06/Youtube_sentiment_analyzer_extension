# YouTube AI Chrome Extension 🚀

A full-stack, machine-learning-powered Chrome Extension built for YouTube.

🔗 **\[Frontend Repository\]**: [Click here to view the Chrome Extension Code](https://github.com/AshokChoudhary06/Youtube-sentiment-analyzer-frontend)
🔗 **\[Backend Repository\]**: *(You are here)* - Contains the FastAPI, ML Ops, and AWS CI/CD Infrastructure.

## 🎥 See it in Action

*(If you have a YouTube video, replace `YOUR_VIDEO_ID` below with your actual YouTube video ID!)*

[![Watch the video](https://img.youtube.com/vi/YOUR_VIDEO_ID/hqdefault.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)


## 🌟 Features

* **Seamless YouTube Integration:** Interacts directly with the YouTube DOM to extract and process data.

* **AI/ML Powered:** Utilizes a custom Machine Learning model to deliver intelligent predictions directly in the browser.

* **Highly Available Backend:** Powered by a containerized FastAPI server running on an AWS Auto Scaling Group behind an Application Load Balancer.

* **Fully Automated CI/CD:** A robust GitHub Actions pipeline that builds, tags, and deploys the latest code to AWS seamlessly.

## 🛠️ Tech Stack & Architecture

### Frontend (Chrome Extension)

* JavaScript (ES6+), HTML, CSS

* Manifest V3 Standard

### Backend & Machine Learning (The API)

* **Framework:** Python, FastAPI, Uvicorn

* **ML Ops:** MLflow (Experiment tracking), DVC (Data Version Control)

* **Containerization:** Docker

### Cloud Infrastructure & DevOps (AWS)

* **Compute:** EC2 (Ubuntu), Auto Scaling Groups (ASG), Launch Templates

* **Networking:** Application Load Balancer (ALB), Target Groups

* **Registry:** Elastic Container Registry (ECR)

* **Deployment:** AWS CodeDeploy, GitHub Actions

## 📊 Machine Learning Experiment Tracking (MLflow)

To ensure the highest quality predictions and reproducible model training, this project heavily utilizes **MLflow** for hyperparameter tuning and model versioning, alongside **DVC** for data tracking.

*(See screenshots of the MLflow tracking server below):*

![MLflow Dashboard View 1](path/to/your/first_image.png)

![MLflow Run Details](path/to/your/second_image.png)

## 📥 How to Install the Chrome Extension

Since this extension is currently in development and not yet on the Chrome Web Store, you can easily install it locally in "Developer Mode".

1. **Download the Frontend Code:**

   * Visit the [**Frontend Repository**](https://github.com/AshokChoudhary06/Youtube-sentiment-analyzer-frontend).

   * Click the green **`<> Code`** button at the top of that repository and select **Download ZIP**.

   * Extract the ZIP file to a folder on your computer.

2. **Open Chrome Extensions:**

   * Open Google Chrome and type `chrome://extensions/` into the URL bar and press Enter.

3. **Enable Developer Mode:**

   * In the top-right corner of the Extensions page, toggle the **Developer mode** switch to **ON**.

4. **Load the Extension:**

   * Click the **Load unpacked** button that appears in the top-left corner.

   * Select the folder containing the frontend extension code (make sure you select the folder containing the `manifest.json` file).

5. **Pin it!**

   * Click the puzzle piece icon 🧩 in your Chrome toolbar and click the pin icon next to the extension to keep it accessible. You are ready to go!

## 💻 Local Development Setup (Backend)

If you want to run the FastAPI backend locally on your machine:

1. Clone this backend repository and navigate to the root folder.

2. Build the Docker image:

   ```
   docker build -t youtube-extension-api .
   
   ```

3. Run the Docker container:

   ```
   docker run -d -p 8000:8000 youtube-extension-api
   
   ```

4. Access the API documentation at `http://localhost:8000/docs`.

## ☁ Production Cloud Architecture

This project is deployed using an enterprise-grade AWS architecture:

1. **GitHub Actions:** Merging code into the `main` branch triggers a workflow that builds a new Docker image and pushes it to **AWS ECR**.

2. **AWS CodeDeploy:** Once the image is pushed, CodeDeploy safely pulls the new image to the live EC2 servers, stops the old container, and starts the new one in seconds.

3. **Auto Scaling & Load Balancing:** The application runs on EC2 instances managed by an **Auto Scaling Group**, ensuring self-healing capabilities. An **Application Load Balancer** securely routes all incoming traffic from the Chrome extension to healthy servers.
