FairLens is an AI-powered ethics platform designed for financial institutions to audit loan approval algorithms for systemic bias. Using advanced statistical metrics and generative AI, FairLens detects "Redlining" proxies and remediates historical bias to ensure regulatory compliance (e.g., GDPR, ECOA).

**Key Features**

Automated Fairness Audit: Calculates the Disparate Impact Ratio (DIR) to determine if protected groups (Gender, Race, Age) are being treated unfairly according to the legal "80% Rule."

Proxy Variable Detection: Identifies hidden "Redlining" variables (like Zip Code or Education) that secretly correlate with protected attributes using Mutual Information scoring.

Generative Synthetic Repair: Uses SMOTENC (Synthetic Minority Over-sampling Technique) to rebalance biased datasets, creating "fair" training data for future AI models.

XAI Scorecard: Provides an Explainability Scorecard with side-by-side visual distributions showing the transition from a biased portfolio to a fair one.

🛠️ Tech Stack

Backend: FastAPI (Python 3.11)

Frontend: Streamlit

AI/ML: Scikit-learn, Imbalanced-learn (SMOTENC)

Visualization: Seaborn, Matplotlib

DevOps: Docker, Docker Compose

**How to Run**

FairLens is fully containerized. You do not need to install Python or any libraries locally—just Docker.

Clone the Repository:

**Bash**

```

git clone https://github.com/yourusername/fairlens-project.git

cd fairlens-project

```

Launch the Microservices:

**Bash**



```

docker compose up --build


```

Access the Platform:

Dashboard : https://fairlens-ui-1025046730042.asia-south1.run.app/


**How It Works (The Math)**

1. Disparate Impact Ratio (DIR)
FairLens evaluates fairness by comparing the probability of a positive outcome (Loan Approved) for the unprivileged group versus the privileged group.

Note: A DIR score below 0.8 indicates a high risk of systemic bias.

2. Proxy Detection
Instead of just looking at gender or race, FairLens uses Mutual Information (MI) to see how much a "neutral" feature (like Experience Years or Credit Limit) reveals about a protected attribute. This prevents "Bias by Proxy."

3. Synthetic Remediation
When bias is detected, the engine doesn't just delete data. It uses Generative AI to create new, mathematically plausible data points for the underrepresented group until the decision boundary is fair. This ensures the resulting model is both accurate and ethical.

The sample dataset is from Kaggle
