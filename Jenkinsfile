// Triggered by a GitHub push webhook (Generic Webhook Trigger plugin).
// Retrains on all data in data/ and tells the API to reload the new model.
pipeline {
    agent any

    environment {
        API_URL = 'http://localhost:5000'
    }

    stages {
        stage('Install dependencies') {
            steps {
                bat 'python -m pip install -q -r requirements.txt'
            }
        }
        stage('Train on Kaggle GPU') {
            steps {
                powershell 'powershell -ExecutionPolicy Bypass -File scripts\\train_on_kaggle.ps1'
            }
        }
        stage('Reload API') {
            steps {
                powershell '''
                    $r = Invoke-RestMethod -Method Post -Uri "$env:API_URL/reload"
                    Write-Output "Reloaded model: accuracy $($r.accuracy)"
                '''
            }
        }
    }

    post {
        success { echo 'Pipeline finished: new model is live.' }
        failure { echo 'Pipeline failed: check the stage logs.' }
    }
}
