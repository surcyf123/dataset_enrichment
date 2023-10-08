# %%
import boto3
import os

def download_s3_bucket(bucket_name, download_path):
    """
    Download all contents of an S3 bucket to the specified local path.
    :param bucket_name: Name of the S3 bucket
    :param download_path: Local path to download the contents
    """
    # Create an S3 client
    s3 = boto3.client('s3',
                aws_access_key_id='AKIAX5ZWWZTUO6ZMIJ4M',
                aws_secret_access_key='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')
    
    
    # List all objects in the bucket
    objects = s3.list_objects_v2(Bucket=bucket_name)
    
    while objects.get('Contents'):
        for obj in objects['Contents']:
            key = obj['Key']
            local_filename = os.path.join(download_path, key)

            # If the key represents a directory in S3
            if key.endswith('/'):
                if not os.path.exists(local_filename):
                    os.makedirs(local_filename)
                continue  # Skip the rest of the loop as there's no file to download for directories

            # Ensure directory exists for the file
            local_directory = os.path.dirname(local_filename)
            if not os.path.exists(local_directory):
                os.makedirs(local_directory)

            # Download the file
            s3.download_file(bucket_name, key, local_filename)


        
        # Check for truncated results and continue
        if objects.get('IsTruncated'):
            continuation_key = objects.get('NextContinuationToken')
            objects = s3.list_objects_v2(Bucket=bucket_name, ContinuationToken=continuation_key)
        else:
            break

if __name__ == '__main__':
    import os
    BUCKET_NAME = 'quantized-language-model-results'
    DOWNLOAD_PATH = 'exp_results'
    download_s3_bucket(BUCKET_NAME, DOWNLOAD_PATH)

# %%
