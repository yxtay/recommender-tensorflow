from google.api_core import exceptions
from google.cloud import bigquery
from google.oauth2 import service_account

from src.logger import get_logger

logger = get_logger(__name__)


def get_credentials(service_account_json):
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials = (service_account.Credentials
                   .from_service_account_file(service_account_json, scopes=scopes))
    logger.info("credentials created from %s.", service_account_json)
    return credentials


def get_bigquery_client(service_account_json):
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials = (service_account.Credentials
                   .from_service_account_file(service_account_json, scopes=scopes))
    logger.info("credentials created from %s.", service_account_json)

    # get client
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)
    return client


def get_bigquery_table(table_id, dataset_id, client):
    # get or create dataset
    try:
        dataset = client.get_dataset(dataset_id)
    except exceptions.NotFound:
        dataset = client.create_dataset(dataset_id)
        logger.info("dataset %s created, since not found.", dataset_id)

    # get table
    table = dataset.table(table_id)
    return table


def df_to_bigquery(df, table_id, dataset_id, client):
    table = get_bigquery_table(table_id, dataset_id, client)

    # set config: insert overwrite
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.job.WriteDisposition.WRITE_TRUNCATE
    )

    # insert table
    job = client.load_table_from_dataframe(
        dataframe=df.compute().rename_axis("id"),
        destination=table,
        job_config=job_config
    )
    job.result()
    logger.info('%s rows loaded into %s.%s.%s.', job.output_rows, job.project, dataset_id, table_id)
    return table


def bigquery_to_table(query, table_id, dataset_id, client):
    table = get_bigquery_table(table_id, dataset_id, client)

    # set config: insert overwrite to table
    job_config = bigquery.QueryJobConfig(
        destination=table,
        write_disposition=bigquery.job.WriteDisposition.WRITE_TRUNCATE
    )

    # query and insert to table
    job = client.query(query, job_config=job_config)
    job.result()
    logger.info("query results loaded to table: %s.", table.path)
    return table


def bigquery_to_gcs(table_id, dataset_id, path, bucket, client):
    table = get_bigquery_table(table_id, dataset_id, client)
    destination_uri = "gs://{bucket}/{path}".format(bucket=bucket, path=path)

    job = client.extract_table(table, destination_uri)
    job.result()
    logger.info("exported %s:%s.%s to %s.",
                client.project, dataset_id, table_id, destination_uri)
