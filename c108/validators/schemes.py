"""
Data Validators Schemes
"""

from ..abc import classgetter


class SchemeGroup:
    """Base class for URI scheme groups."""

    @classgetter(cache=True)
    def all(cls) -> tuple[str, ...]:
        """
        Get all schemes in this group.

        Append all schemes from nested SchemeGroup instances recursively
        """
        schemes = []
        for attr_name in dir(cls):
            if not attr_name.startswith("_") and attr_name != "all":
                attr = getattr(cls, attr_name)
                if isinstance(attr, str):
                    schemes.append(attr)
                elif isinstance(attr, type) and issubclass(attr, SchemeGroup):
                    schemes.extend(attr.all)
        return tuple(schemes)


class AnalyticalSchemes(SchemeGroup):
    """Analytical/OLAP database URI schemes."""

    clickhouse = "clickhouse"
    databricks = "databricks"
    druid = "druid"
    impala = "impala"
    presto = "presto"
    snowflake = "snowflake"
    trino = "trino"
    vertica = "vertica"


class AWSDatabaseSchemes(SchemeGroup):
    """AWS managed database URI schemes."""

    athena = "athena"  # Serverless query service
    aurora = "aurora"  # Aurora MySQL/PostgreSQL
    documentdb = "documentdb"  # MongoDB-compatible
    dynamodb = "dynamodb"  # NoSQL key-value
    neptune_db = "neptune-db"  # Graph database
    rds = "rds"  # Relational Database Service
    redshift = "redshift"  # Data warehouse
    timestream = "timestream"  # Time series database


class AWSStorageSchemes(SchemeGroup):
    """AWS S3 URI schemes."""

    s3 = "s3"
    s3a = "s3a"
    s3n = "s3n"


class AzureDatabaseSchemes(SchemeGroup):
    """Azure managed database URI schemes."""

    azuresql = "azuresql"  # Azure SQL Database
    cosmosdb = "cosmosdb"  # Multi-model NoSQL
    sqldw = "sqldw"  # SQL Data Warehouse (legacy name)
    synapse = "synapse"  # Analytics platform (formerly SQL DW)


class AzureStorageSchemes(SchemeGroup):
    """Microsoft Azure storage URI schemes."""

    abfs = "abfs"
    abfss = "abfss"
    adl = "adl"
    az = "az"
    wasb = "wasb"
    wasbs = "wasbs"


class DataVersioningSchemes(SchemeGroup):
    """Data versioning system URI schemes."""

    dvc = "dvc"  # DVC (Data Version Control)
    pachyderm = "pachyderm"  # Pachyderm data pipelines


class DistributedSchemes(SchemeGroup):
    """Distributed file system URI schemes."""

    alluxio = "alluxio"
    ceph = "ceph"
    dbfs = "dbfs"
    minio = "minio"
    rados = "rados"
    swift = "swift"


class GCPDatabaseSchemes(SchemeGroup):
    """GCP managed database URI schemes."""

    bigquery = "bigquery"  # Data warehouse
    bigtable = "bigtable"  # NoSQL wide-column
    datastore = "datastore"  # NoSQL document database (legacy)
    firestore = "firestore"  # NoSQL document database
    spanner = "spanner"  # Distributed SQL database


class GCPStorageSchemes(SchemeGroup):
    """Google Cloud Platform URI schemes."""

    gs = "gs"


class GraphSchemes(SchemeGroup):
    """Graph database URI schemes."""

    arangodb = "arangodb"
    janusgraph = "janusgraph"
    neo4j = "neo4j"
    neo4js = "neo4js"  # Neo4j with encryption
    orientdb = "orientdb"


class HadoopSchemes(SchemeGroup):
    """Hadoop ecosystem URI schemes."""

    hdfs = "hdfs"
    hive = "hive"
    webhdfs = "webhdfs"


class LakehouseSchemes(SchemeGroup):
    """Data lakehouse URI schemes."""

    delta = "delta"
    iceberg = "iceberg"


class LocalSchemes(SchemeGroup):
    """Local and URN schemes."""

    file = "file"
    urn = "urn"


class MLDatasetSchemes(SchemeGroup):
    """ML dataset URI schemes."""

    tfds = "tfds"  # TensorFlow Datasets
    torch = "torch"  # PyTorch datasets


class MLFlowSchemes(SchemeGroup):
    """MLflow-specific URI schemes."""

    models = "models"  # Model Registry: models:/<name>/<version_or_stage>
    runs = "runs"  # Artifact from run: runs:/<run_id>/path


class MLHubSchemes(SchemeGroup):
    """ML model hub URI schemes."""

    hf = "hf"  # Hugging Face Hub
    huggingface = "huggingface"  # Hugging Face Hub (alias)
    onnx = "onnx"  # ONNX Model Zoo
    tfhub = "tfhub"  # TensorFlow Hub
    torchhub = "torchhub"  # PyTorch Hub


class MLTrackingSchemes(SchemeGroup):
    """ML experiment tracking platform URI schemes."""

    aim = "aim"  # Aim
    clearml = "clearml"  # ClearML (formerly Allegro)
    comet = "comet"  # Comet ML
    mlflow = "mlflow"  # MLflow artifacts (generic)
    neptune = "neptune"  # Neptune.ai
    sacred = "sacred"  # Sacred
    tensorboard = "tensorboard"  # TensorBoard logs
    wandb = "wandb"  # Weights & Biases


class NetworkFSSchemes(SchemeGroup):
    """Network file system URI schemes."""

    afp = "afp"
    cifs = "cifs"
    nfs = "nfs"
    smb = "smb"


class NoSQLSchemes(SchemeGroup):
    """NoSQL database URI schemes."""

    cassandra = "cassandra"
    couchbase = "couchbase"
    couchdb = "couchdb"
    cql = "cql"  # Cassandra Query Language
    memcached = "memcached"
    mongo = "mongo"  # Alternative MongoDB scheme
    mongodb = "mongodb"
    redis = "redis"
    rediss = "rediss"  # Redis with SSL/TLS


class SearchSchemes(SchemeGroup):
    """Search and vector database URI schemes."""

    elasticsearch = "elasticsearch"
    es = "es"  # Elasticsearch alias
    meilisearch = "meilisearch"
    opensearch = "opensearch"
    solr = "solr"
    typesense = "typesense"


class SQLSchemes(SchemeGroup):
    """SQL database URI schemes."""

    cockroach = "cockroach"
    cockroachdb = "cockroachdb"
    db2 = "db2"
    mariadb = "mariadb"
    mssql = "mssql"
    mysql = "mysql"
    oracle = "oracle"
    postgres = "postgres"
    postgresql = "postgresql"
    sqlite = "sqlite"
    sqlserver = "sqlserver"
    teradata = "teradata"


class TimeSeriesSchemes(SchemeGroup):
    """Time series database URI schemes."""

    influxdb = "influxdb"
    prometheus = "prometheus"
    timescaledb = "timescaledb"
    victoriametrics = "victoriametrics"


class VectorSchemes(SchemeGroup):
    """Vector database URI schemes (for ML embeddings)."""

    chroma = "chroma"
    chromadb = "chromadb"
    milvus = "milvus"
    pinecone = "pinecone"
    qdrant = "qdrant"
    weaviate = "weaviate"


class WebSchemes(SchemeGroup):
    """Web protocol URI schemes."""

    ftp = "ftp"
    ftps = "ftps"
    http = "http"
    https = "https"


class Scheme:
    """URI scheme definitions organized by category.

    Provides categorized access to all supported URI schemes for cloud storage,
    distributed systems, ML platforms, experiment tracking, and databases.

    Examples:
        >>> # Cloud storage
        >>> Scheme.aws.s3
        's3'

        >>> # ML experiment tracking
        >>> Scheme.ml.tracking.wandb
        'wandb'

        >>> # MLflow-specific
        >>> Scheme.ml.mlflow.runs
        'runs'

        >>> # Model hubs
        >>> Scheme.ml.hub.hf
        'hf'

        >>> # Cloud databases
        >>> Scheme.db.cloud.aws.bigquery
        Traceback (most recent call last):
        ...
        AttributeError: type object 'AWSDatabaseSchemes' has no attribute 'bigquery'

        >>> # Cloud databases (corrected)
        >>> Scheme.db.cloud.gcp.bigquery
        'bigquery'

        >>> # Vector databases
        >>> Scheme.db.vector.pinecone
        'pinecone'

        >>> # Get all database schemes
        >>> schemes = Scheme.db.all
        >>> 'bigquery' in schemes and 'redis' in schemes
        True
    """

    # Cloud providers (storage)
    aws = AWSStorageSchemes
    azure = AzureStorageSchemes
    gcp = GCPStorageSchemes

    # Distributed systems
    distributed = DistributedSchemes
    hadoop = HadoopSchemes
    lakehouse = LakehouseSchemes
    network = NetworkFSSchemes

    # ML/AI platforms (nested for organization)
    class ml:
        """ML/AI platform schemes organized by category."""

        data_versioning = DataVersioningSchemes
        datasets = MLDatasetSchemes
        hub = MLHubSchemes
        mlflow = MLFlowSchemes
        tracking = MLTrackingSchemes

        @classgetter(cache=True)
        def all(cls) -> tuple[str, ...]:
            """Get all ML-related schemes.

            Returns:
                tuple[str, ...]: All ML platform, tracking, hub, and dataset schemes.

            Examples:
                >>> schemes = Scheme.ml.all
                >>> 'wandb' in schemes and 'hf' in schemes and 'runs' in schemes
                True
            """
            return (
                *DataVersioningSchemes.all,
                *MLDatasetSchemes.all,
                *MLFlowSchemes.all,
                *MLHubSchemes.all,
                *MLTrackingSchemes.all,
            )

    # Databases (comprehensive organization)
    class db:
        """Database schemes organized by category."""

        # SQL databases
        sql = SQLSchemes

        # Cloud-managed databases
        class cloud:
            """Cloud-managed database schemes."""

            aws = AWSDatabaseSchemes
            azure = AzureDatabaseSchemes
            gcp = GCPDatabaseSchemes

            @classgetter(cache=True)
            def all(cls) -> tuple[str, ...]:
                """Get all cloud-managed database schemes."""
                return (
                    *AWSDatabaseSchemes.all,
                    *AzureDatabaseSchemes.all,
                    *GCPDatabaseSchemes.all,
                )

        # Database types
        analytical = AnalyticalSchemes
        graph = GraphSchemes
        nosql = NoSQLSchemes
        search = SearchSchemes
        timeseries = TimeSeriesSchemes
        vector = VectorSchemes

        @classgetter(cache=True)
        def all(cls) -> tuple[str, ...]:
            """Get all database schemes.

            Returns:
                tuple[str, ...]: All database schemes including cloud, NoSQL,
                    vector, time series, graph, analytical, and sql databases.

            Examples:
                >>> schemes = Scheme.db.all
                >>> all(s in schemes for s in ['bigquery', 'redis', 'pinecone', 'neo4j'])
                True
            """
            return (
                *AWSDatabaseSchemes.all,
                *AnalyticalSchemes.all,
                *AzureDatabaseSchemes.all,
                *GCPDatabaseSchemes.all,
                *GraphSchemes.all,
                *NoSQLSchemes.all,
                *SQLSchemes.all,
                *SearchSchemes.all,
                *TimeSeriesSchemes.all,
                *VectorSchemes.all,
            )

    # Web and local
    local = LocalSchemes
    web = WebSchemes

    @staticmethod
    def cloud() -> tuple[str, ...]:
        """Get all major cloud provider schemes (AWS, GCP, Azure storage)."""
        return (
            *AWSStorageSchemes.all,
            *AzureStorageSchemes.all,
            *GCPStorageSchemes.all,
        )

    @staticmethod
    def bigdata() -> tuple[str, ...]:
        """Get all big data / distributed system schemes."""
        return (
            *DistributedSchemes.all,
            *HadoopSchemes.all,
            *LakehouseSchemes.all,
        )

    @classgetter(cache=True)
    def all(cls) -> tuple[str, ...]:
        """Get all supported URI schemes."""
        return (
            *AWSDatabaseSchemes.all,
            *AWSStorageSchemes.all,
            *AnalyticalSchemes.all,
            *AzureDatabaseSchemes.all,
            *AzureStorageSchemes.all,
            *DataVersioningSchemes.all,
            *DistributedSchemes.all,
            *GCPDatabaseSchemes.all,
            *GCPStorageSchemes.all,
            *GraphSchemes.all,
            *HadoopSchemes.all,
            *LakehouseSchemes.all,
            *LocalSchemes.all,
            *MLDatasetSchemes.all,
            *MLFlowSchemes.all,
            *MLHubSchemes.all,
            *MLTrackingSchemes.all,
            *NetworkFSSchemes.all,
            *NoSQLSchemes.all,
            *SQLSchemes.all,
            *SearchSchemes.all,
            *TimeSeriesSchemes.all,
            *VectorSchemes.all,
            *WebSchemes.all,
        )
