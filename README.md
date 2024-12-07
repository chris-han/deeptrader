ALGORITHMIC TRADING Pratice
yahoo_fin as data reader
backtrader as strategy engine
sklearn as predition model

``` mermaid
flowchart TB
    subgraph "元数据采集与血缘追踪层"
        MetaCollector[元数据智能采集器<br/>Apache Atlas/DataHub]
        LineageTracker[数据血缘追踪引擎<br/>OpenLineage/Marquez]
        AIAgent[元数据管理AI智能体<br/>LlamaIndex/Haystack]
    end

    subgraph "数据资产管理层"
        AssetCatalog[数据资产目录<br/>OpenMetadata/Amundsen]
        AssetClassifier[数据分类标签引擎<br/>DataHub/Apache Atlas]
        StandardDefiner[数据标准定义模块<br/>Apache Sentinel/Deequ]
    end

    subgraph "数据质量监控层"
        QualityMonitor[智能质量监控中心<br/>Great Expectations/Deequ]
        AnomalyDetector[异常数据智能发现<br/>Alibi Detect/PyOD]
        WorkorderGenerator[问题工单生成器<br/>Apache Airflow/Prefect]
    end

    subgraph "外部数据集成层"
        ExternalDataIntegrator[外部数据智能集成器<br/>Apache Nifi/Airbyte]
        MarketAnalyzer[市场分析报告生成器<br/>Pandas/PySpark]
    end

    subgraph "大模型技术支撑"
        LocalLLM[本地大语言模型<br/>Llama3.2/Ollama]
        EmbeddingService[向量嵌入服务<br/>Sentence Transformers]
        RetrievalService[语义检索服务<br/>Faiss/Elastic]
    end

    MetaCollector --> LineageTracker
    MetaCollector --> AssetClassifier
    LineageTracker --> AssetCatalog
    AssetClassifier --> AssetCatalog
    StandardDefiner --> AssetCatalog

    QualityMonitor --> AnomalyDetector
    AnomalyDetector --> WorkorderGenerator

    ExternalDataIntegrator --> MarketAnalyzer
    MarketAnalyzer --> AssetCatalog

    LocalLLM --> MetaCollector
    LocalLLM --> AssetClassifier
    LocalLLM --> QualityMonitor
    LocalLLM --> ExternalDataIntegrator
```

``` mermaid
sequenceDiagram
    participant Dev as Developer
    participant SCM as Source Control (Git)
    participant Jenkins as Jenkins Pipeline
    participant SonarQube as SonarQube
    participant TestFramework as Test Framework
    participant ArtifactRepo as Artifact Repository
    participant K8s as Kubernetes
    participant Monitor as Monitoring System

    Dev->>SCM: Push Code
    SCM->>Jenkins: Trigger Build
    
    Jenkins->>Jenkins: Code Checkout
    Note right of Jenkins: Plugins Used:<br/>- Git Plugin<br/>- Pipeline Plugin

    Jenkins->>SonarQube: Static Code Analysis
    Note right of Jenkins: Plugins Used:<br/>- SonarQube Scanner Plugin<br/>- Quality Gates

    Jenkins->>TestFramework: Run Unit Tests
    Note right of Jenkins: Plugins Used:<br/>- Python Plugin<br/>- pytest<br/>- Coverage Plugin

    Jenkins->>Jenkins: Build Python Package
    Note right of Jenkins: Plugins Used:<br/>- Python Packaging Plugin<br/>- Wheel/Setuptools

    Jenkins->>ArtifactRepo: Push Artifact
    Note right of Jenkins: Plugins Used:<br/>- Artifactory Plugin<br/>- Nexus Plugin

    Jenkins->>K8s: Deploy to Kubernetes
    Note right of Jenkins: Plugins Used:<br/>- Kubernetes Deployment Plugin<br/>- Cloud Foundry Plugin

    K8s->>Monitor: Send Metrics
    Note right of Jenkins: Plugins Used:<br/>- Prometheus Plugin<br/>- Grafana Plugin
```
``` mermaid
sequenceDiagram
    participant App as Application
    participant AKS as Azure Kubernetes Service
    participant ContainerInsights as Azure Container Insights
    participant LogAnalytics as Azure Log Analytics
    participant AppInsights as Azure Application Insights
    participant Monitor as Azure Monitor
    participant Grafana as Grafana Dashboard
    participant AlertSystem as Alert Management

    App->>AKS: Generates Logs/Metrics
    AKS->>ContainerInsights: Collect Cluster Metrics
    
    ContainerInsights->>LogAnalytics: Store Cluster Logs
    App->>AppInsights: Send Application Telemetry
    
    LogAnalytics->>Monitor: Centralize Logs
    AppInsights->>Monitor: Push Application Metrics
    
    Monitor->>Grafana: Stream Visualization Data
    Monitor->>AlertSystem: Trigger Alerts
    
    AlertSystem->>AlertSystem: Evaluate Alert Conditions
    AlertSystem->>Slack: Send Notifications
    AlertSystem->>PagerDuty: Escalate Critical Alerts
```
#### CI/CD pipeline design with github acition + dokploy + aws ec2,
```mermaid
sequenceDiagram
    participant Dev as Developer
    participant GH as GitHub Repository
    participant GA as GitHub Actions
    participant DP as Dokploy
    participant EC2 as AWS EC2
    participant Mon as Monitoring

    Dev->>GH: Push code to repository
    GH->>GA: Trigger CI/CD workflow
    GA->>GA: Run tests
    GA->>GA: Build Docker image
    GA->>DP: Deploy image to Dokploy
    DP->>EC2: Deploy application
    EC2->>Mon: Send health metrics
    Mon->>Dev: Notify deployment status
```
#### GitHub Actions workflow script:
``` script
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  DOKPLOY_TOKEN: ${{ secrets.DOKPLOY_TOKEN }}
  DOKPLOY_PROJECT_ID: ${{ secrets.DOKPLOY_PROJECT_ID }}
  AWS_REGION: us-east-1

jobs:
  test-and-build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    # Set up Node.js (adjust based on your project)
    - name: Use Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20.x'
    
    # Install dependencies
    - name: Install Dependencies
      run: npm ci
    
    # Run tests
    - name: Run Tests
      run: npm test
    
    # Build Docker image
    - name: Build Docker Image
      run: docker build -t my-app:${{ github.sha }} .
    
    # Login to Docker registry
    - name: Login to Docker Registry
      run: echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
    
    # Push image to registry
    - name: Push to Docker Registry
      run: |
        docker tag my-app:${{ github.sha }} ${{ secrets.DOCKER_REGISTRY }}/my-app:${{ github.sha }}
        docker push ${{ secrets.DOCKER_REGISTRY }}/my-app:${{ github.sha }}

  deploy:
    needs: test-and-build
    runs-on: ubuntu-latest
    
    steps:
    # Deploy to Dokploy
    - name: Deploy to Dokploy
      run: |
        curl -X POST \
          -H "Authorization: Bearer $DOKPLOY_TOKEN" \
          -H "Content-Type: application/json" \
          -d '{
            "projectId": "'$DOKPLOY_PROJECT_ID'",
            "image": "'${{ secrets.DOCKER_REGISTRY }}/my-app:${{ github.sha }}'"
          }' \
          https://dokploy.yourdomain.com/api/deployments
    
    # Optional: Health check
    - name: Check Deployment Health
      run: |
        sleep 60  # Wait for deployment
        curl -f https://your-app-domain.com/health || exit 1
```
#### Dokploy configuration script:
```script
version: '3.8'

services:
  app:
    image: ${DOCKER_REGISTRY}/my-app:${GITHUB_SHA}
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=${DATABASE_URL}
    deploy:
      replicas: 2
      update_config:
        parallelism: 1
        delay: 10s
        order: stop-first
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

networks:
  app_network:
    driver: bridge
```

####  EC2 setup script:
```
#!/bin/bash

# Update system
sudo yum update -y

# Install Docker
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install CloudWatch Agent for monitoring
sudo yum install -y amazon-cloudwatch-agent

# Configure CloudWatch Agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard

# Start CloudWatch Agent
sudo systemctl start amazon-cloudwatch-agent
sudo systemctl enable amazon-cloudwatch-agent

# Pull and run application (typically done via Dokploy)
docker pull ${DOCKER_REGISTRY}/my-app:latest
docker run -d -p 3000:3000 ${DOCKER_REGISTRY}/my-app:latest
```