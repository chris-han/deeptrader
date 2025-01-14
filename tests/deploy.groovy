/* groovylint-disable LineLength */
/* groovylint-disable-next-line MethodReturnTypeRequired, NoDef */
def deployToDokploy() {
    withCredentials([string(credentialsId: 'dokploy-token', variable: 'DOKPLOY_TOKEN')]) {
        sh """
            curl -X POST \
            -H "Authorization: Bearer ${DOKPLOY_TOKEN}" \
            -H "Content-Type: application/json" \
            -d '{
                "projectId": "${env.DOKPLOY_PROJECT_ID}",
                "image": "${DOCKER_REGISTRY}/${DOCKER_IMAGE_NAME}:${env.DOCKER_IMAGE_TAG}"
            }' \
            https://dokploy.mightytech.cn/api/deployments
        """
    }
}

/* groovylint-disable-next-line CompileStatic */
pipeline {
    agent any

    environment {
        DOCKER_IMAGE_NAME = 'deeptrader'
        DOCKER_REGISTRY = 'rongqitest.azurecr.cn'
        AZURE_CREDS = credentials('azure-registry-credentials')
    }

    stages {
        stage('Checkout') {
            steps {
                git url: 'https://github.com/chris-han/deeptrader.git', branch: 'main'
            }
        }

        stage('Build') {
            steps {
                script {
                    /* groovylint-disable-next-line NoDef, VariableTypeRequired */
                    def dockerImageTag = "${env.BUILD_NUMBER}-${env.GIT_COMMIT.substring(0, 7)}"

                    // Login to Azure registry
                    sh "docker login ${DOCKER_REGISTRY} -u ${AZURE_CREDS_USR} -p ${AZURE_CREDS_PSW}"

                    echo "Building Docker image ${DOCKER_IMAGE_NAME}:${dockerImageTag}"
                    sh "docker build -t ${DOCKER_IMAGE_NAME}:${dockerImageTag} ."
                    /* groovylint-disable-next-line NoDef, VariableTypeRequired */
                    def tagCommand = "docker tag ${DOCKER_IMAGE_NAME}:${dockerImageTag} " +
                        "${DOCKER_REGISTRY}/${DOCKER_IMAGE_NAME}:${dockerImageTag}"
                    sh tagCommand
                    sh "docker push ${DOCKER_REGISTRY}/${DOCKER_IMAGE_NAME}:${dockerImageTag}"

                    env.DOCKER_IMAGE_TAG = dockerImageTag
                }
            }
        }

        stage('Test') {
            steps {
                    script {
                            echo 'Running Tests'

                            sh 'python -m pytest --cov=./ --cov-report=xml'

                    // Assuming you want coverage reports, you can add steps for generating & publishing this here
                    }
            }
        }

        stage('Publish Coverage Report') {
            steps {
                publishCoverage adapters: [coberturaAdapter('coverage.xml')]
            }
        }

        stage('Deploy to Dokploy') {
            steps {
                deployToDokploy()
            }
        }

        stage('Health Check') {
            steps {
                script {
                    sleep(60)  // Wait for deployment to be available
                    sh 'curl -f https://deeptrader.mightytech.cn/health || exit 1'
                }
            }
        }
    }
    post {
        always {
            sh "docker logout ${DOCKER_REGISTRY}"
            cleanWs()
        }
    }
}
