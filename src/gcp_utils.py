from dask import dataframe as dd
from google.cloud import bigquery
from google.api_core import exceptions
from google.oauth2 import service_account

from src.logger import get_logger

logger = get_logger(__name__)


def get_credentials(service_account_json):
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials = (service_account.Credentials
                   .from_service_account_file(service_account_json, scopes=scopes))
    return credentials


def df_upload_bigquery(df: dd.DataFrame, dataset_id: str, table_id: str, credentials=None):
    # get client
    client = bigquery.Client(credentials=credentials)

    # get or create dataset
    try:
        dataset = client.get_dataset(dataset_id)
    except exceptions.NotFound:
        dataset = client.create_dataset(dataset_id)
        logger.info("dataset %s created, since not found.", dataset_id)

    # get table
    table = dataset.table(table_id)
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
    # wait for job to complete
    job.result()
    logger.info('loaded %s rows into %s.%s.%s.', job.output_rows, job.project, dataset_id, table_id)
    return table
