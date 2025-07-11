# import boto3
# import os
# from botocore.exceptions import NoCredentialsError

# S3_BUCKET = os.getenv("S3_BUCKET")
# S3_ENDPOINT = os.getenv("S3_ENDPOINT")
# S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
# S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")

# session = boto3.session.Session()
# s3 = session.client(
#     service_name='s3',
#     endpoint_url=S3_ENDPOINT,
#     aws_access_key_id=S3_ACCESS_KEY,
#     aws_secret_access_key=S3_SECRET_KEY
# )

# def upload_to_s3(local_path, s3_key):
#     try:
#         s3.upload_file(local_path, S3_BUCKET, s3_key)
#         return True
#     except NoCredentialsError:
#         return False

# def download_from_s3(s3_key, local_path):
#     try:
#         s3.download_file(S3_BUCKET, s3_key, local_path)
#         return True
#     except NoCredentialsError:
#         return False

import boto3
import os
from botocore.exceptions import NoCredentialsError, ClientError

S3_BUCKET = os.getenv("S3_BUCKET")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")

session = boto3.session.Session()
s3 = session.client(
    service_name='s3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

def upload_to_s3(local_path, s3_key):
    try:
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        print(f"Successfully uploaded {local_path} to {s3_key}")
        return True
    except (NoCredentialsError, ClientError) as e:
        print(f"Error uploading to S3: {e}")
        return False

def download_from_s3(s3_key, local_path):
    try:
        # Убедимся, что директория для сохранения файла существует
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(S3_BUCKET, s3_key, local_path)
        print(f"Successfully downloaded {s3_key} to {local_path}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"Error: The object {s3_key} does not exist in bucket {S3_BUCKET}.")
        else:
            print(f"An unexpected error occurred downloading from S3: {e}")
        return False
    except NoCredentialsError as e:
        print(f"Credentials not available: {e}")
        return False