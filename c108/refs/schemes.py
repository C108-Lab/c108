"""
URI scheme constants for databases, cloud storage, distributed systems, and ML platforms.

Provides structured string constants via SchemeBase subclasses and the top-level
Schemes namespace, organized by technology category. Useful for URI parsing,
validation, and routing logic in data pipelines and ML infrastructure.

Classes:
    SchemeBase  – base class with recursive .all collection
    Schemes       – top-level namespace (Schemes.aws.s3, Schemes.db.vector.pinecone, ...)
"""

from ..abc import classgetter


class SchemeBase:
    """Base class for URI scheme groups."""

    @classgetter(cache=True)
    def all(cls) -> tuple[str, ...]:
        """
        Get all schemes in this group.

        Append all schemes from nested SchemeBase instances recursively
        """
        schemes = []
        for attr_name in dir(cls):
            if not attr_name.startswith("_") and attr_name != "all":
                attr = getattr(cls, attr_name)
                if isinstance(attr, str):
                    schemes.append(attr)
                elif isinstance(attr, type) and issubclass(attr, SchemeBase):
                    schemes.extend(attr.all)
        return tuple(schemes)


class Analytical(SchemeBase):
    """Analytical/OLAP database URI schemes."""

    clickhouse = "clickhouse"
    databricks = "databricks"
    druid = "druid"
    impala = "impala"
    presto = "presto"
    snowflake = "snowflake"
    trino = "trino"
    vertica = "vertica"


class AWSDatabase(SchemeBase):
    """AWS managed database URI schemes."""

    athena = "athena"  # Serverless query service
    aurora = "aurora"  # Aurora MySQL/PostgreSQL
    documentdb = "documentdb"  # MongoDB-compatible
    dynamodb = "dynamodb"  # NoSQL key-value
    neptune_db = "neptune-db"  # Graph database
    rds = "rds"  # Relational Database Service
    redshift = "redshift"  # Data warehouse
    timestream = "timestream"  # Time series database


class AWSStorage(SchemeBase):
    """AWS S3 URI schemes."""

    s3 = "s3"
    s3a = "s3a"
    s3n = "s3n"


class AzureDatabase(SchemeBase):
    """Azure managed database URI schemes."""

    azuresql = "azuresql"  # Azure SQL Database
    cosmosdb = "cosmosdb"  # Multi-model NoSQL
    sqldw = "sqldw"  # SQL Data Warehouse (legacy name)
    synapse = "synapse"  # Analytics platform (formerly SQL DW)


class AzureStorage(SchemeBase):
    """Microsoft Azure storage URI schemes."""

    abfs = "abfs"
    abfss = "abfss"
    adl = "adl"
    az = "az"
    wasb = "wasb"
    wasbs = "wasbs"


class DataVersioning(SchemeBase):
    """Data versioning system URI schemes."""

    dvc = "dvc"  # DVC (Data Version Control)
    pachyderm = "pachyderm"  # Pachyderm data pipelines


class Distributed(SchemeBase):
    """Distributed file system URI schemes."""

    alluxio = "alluxio"
    ceph = "ceph"
    dbfs = "dbfs"
    minio = "minio"
    rados = "rados"
    swift = "swift"


class GCPDatabase(SchemeBase):
    """GCP managed database URI schemes."""

    bigquery = "bigquery"  # Data warehouse
    bigtable = "bigtable"  # NoSQL wide-column
    datastore = "datastore"  # NoSQL document database (legacy)
    firestore = "firestore"  # NoSQL document database
    spanner = "spanner"  # Distributed SQL database


class GCPStorage(SchemeBase):
    """Google Cloud Platform URI schemes."""

    gs = "gs"


class Graph(SchemeBase):
    """Graph database URI schemes."""

    arangodb = "arangodb"
    janusgraph = "janusgraph"
    neo4j = "neo4j"
    neo4js = "neo4js"  # Neo4j with encryption
    orientdb = "orientdb"


class Hadoop(SchemeBase):
    """Hadoop ecosystem URI schemes."""

    hdfs = "hdfs"
    hive = "hive"
    webhdfs = "webhdfs"


class Lakehouse(SchemeBase):
    """Data lakehouse URI schemes."""

    delta = "delta"
    iceberg = "iceberg"


class Local(SchemeBase):
    """Local and URN schemes."""

    file = "file"
    urn = "urn"


class MLDataset(SchemeBase):
    """ML dataset URI schemes."""

    tfds = "tfds"  # TensorFlow Datasets
    torch = "torch"  # PyTorch datasets


class MLFlow(SchemeBase):
    """MLflow-specific URI schemes."""

    models = "models"  # Model Registry: models:/<name>/<version_or_stage>
    runs = "runs"  # Artifact from run: runs:/<run_id>/path


class MLHub(SchemeBase):
    """ML model hub URI schemes."""

    hf = "hf"  # Hugging Face Hub
    huggingface = "huggingface"  # Hugging Face Hub (alias)
    onnx = "onnx"  # ONNX Model Zoo
    tfhub = "tfhub"  # TensorFlow Hub
    torchhub = "torchhub"  # PyTorch Hub


class MLTracking(SchemeBase):
    """ML experiment tracking platform URI schemes."""

    aim = "aim"  # Aim
    clearml = "clearml"  # ClearML (formerly Allegro)
    comet = "comet"  # Comet ML
    mlflow = "mlflow"  # MLflow artifacts (generic)
    neptune = "neptune"  # Neptune.ai
    sacred = "sacred"  # Sacred
    tensorboard = "tensorboard"  # TensorBoard logs
    wandb = "wandb"  # Weights & Biases


class NetworkFS(SchemeBase):
    """Network file system URI schemes."""

    afp = "afp"
    cifs = "cifs"
    nfs = "nfs"
    smb = "smb"


class NoSQL(SchemeBase):
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


class Search(SchemeBase):
    """Search and vector database URI schemes."""

    elasticsearch = "elasticsearch"
    es = "es"  # Elasticsearch alias
    meilisearch = "meilisearch"
    opensearch = "opensearch"
    solr = "solr"
    typesense = "typesense"


class SQL(SchemeBase):
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


class TimeSeries(SchemeBase):
    """Time series database URI schemes."""

    influxdb = "influxdb"
    prometheus = "prometheus"
    timescaledb = "timescaledb"
    victoriametrics = "victoriametrics"


class Vector(SchemeBase):
    """Vector database URI schemes (for ML embeddings)."""

    chroma = "chroma"
    chromadb = "chromadb"
    milvus = "milvus"
    pinecone = "pinecone"
    qdrant = "qdrant"
    weaviate = "weaviate"


class Web(SchemeBase):
    """Web protocol URI schemes."""

    ftp = "ftp"
    ftps = "ftps"
    http = "http"
    https = "https"


class Schemes:
    """URI scheme definitions organized by category.

    Provides categorized access to all supported URI schemes for cloud storage,
    distributed systems, ML platforms, experiment tracking, and databases.

    Examples:
        >>> # Cloud storage
        >>> Schemes.aws.s3
        's3'

        >>> # ML experiment tracking
        >>> Schemes.ml.tracking.wandb
        'wandb'

        >>> # MLflow-specific
        >>> Schemes.ml.mlflow.runs
        'runs'

        >>> # Model hubs
        >>> Schemes.ml.hub.hf
        'hf'

        >>> # Cloud databases
        >>> Schemes.db.cloud.aws.bigquery
        Traceback (most recent call last):
        ...
        AttributeError: type object 'AWSDatabase' has no attribute 'bigquery'

        >>> # Cloud databases (corrected)
        >>> Schemes.db.cloud.gcp.bigquery
        'bigquery'

        >>> # Vector databases
        >>> Schemes.db.vector.pinecone
        'pinecone'

        >>> # Get all database schemes
        >>> schemes = Schemes.db.all
        >>> 'bigquery' in schemes and 'redis' in schemes
        True
    """

    # Cloud providers (storage)
    aws = AWSStorage
    azure = AzureStorage
    gcp = GCPStorage

    # Distributed systems
    distributed = Distributed
    hadoop = Hadoop
    lakehouse = Lakehouse
    network = NetworkFS

    # ML/AI platforms (nested for organization)
    class ml:
        """ML/AI platform schemes organized by category."""

        data_versioning = DataVersioning
        datasets = MLDataset
        hub = MLHub
        mlflow = MLFlow
        tracking = MLTracking

        @classgetter(cache=True)
        def all(cls) -> tuple[str, ...]:
            """Get all ML-related schemes.

            Returns:
                tuple[str, ...]: All ML platform, tracking, hub, and dataset schemes.

            Examples:
                >>> schemes = Schemes.ml.all
                >>> 'wandb' in schemes and 'hf' in schemes and 'runs' in schemes
                True
            """
            return (
                *DataVersioning.all,
                *MLDataset.all,
                *MLFlow.all,
                *MLHub.all,
                *MLTracking.all,
            )

    # Databases (comprehensive organization)
    class db:
        """Database schemes organized by category."""

        # SQL databases
        sql = SQL

        # Cloud-managed databases
        class cloud:
            """Cloud-managed database schemes."""

            aws = AWSDatabase
            azure = AzureDatabase
            gcp = GCPDatabase

            @classgetter(cache=True)
            def all(cls) -> tuple[str, ...]:
                """Get all cloud-managed database schemes."""
                return (
                    *AWSDatabase.all,
                    *AzureDatabase.all,
                    *GCPDatabase.all,
                )

        # Database types
        analytical = Analytical
        graph = Graph
        nosql = NoSQL
        search = Search
        timeseries = TimeSeries
        vector = Vector

        # Class Getters ----------------------------------------------------------------------------

        @classgetter(cache=True)
        def all(cls) -> tuple[str, ...]:
            """Get all database schemes.

            Returns:
                tuple[str, ...]: All database schemes including cloud, NoSQL,
                    vector, time series, graph, analytical, and sql databases.

            Examples:
                >>> schemes = Schemes.db.all
                >>> all(s in schemes for s in ['bigquery', 'redis', 'pinecone', 'neo4j'])
                True
            """
            return (
                *AWSDatabase.all,
                *Analytical.all,
                *AzureDatabase.all,
                *GCPDatabase.all,
                *Graph.all,
                *NoSQL.all,
                *SQL.all,
                *Search.all,
                *TimeSeries.all,
                *Vector.all,
            )

    # Web and local
    local = Local
    web = Web

    @classgetter(cache=True)
    def all(cls) -> tuple[str, ...]:
        """Get all supported URI schemes."""
        return (
            *AWSDatabase.all,
            *AWSStorage.all,
            *Analytical.all,
            *AzureDatabase.all,
            *AzureStorage.all,
            *DataVersioning.all,
            *Distributed.all,
            *GCPDatabase.all,
            *GCPStorage.all,
            *Graph.all,
            *Hadoop.all,
            *Lakehouse.all,
            *Local.all,
            *MLDataset.all,
            *MLFlow.all,
            *MLHub.all,
            *MLTracking.all,
            *NetworkFS.all,
            *NoSQL.all,
            *SQL.all,
            *Search.all,
            *TimeSeries.all,
            *Vector.all,
            *Web.all,
        )

    @classgetter(cache=True)
    def bigdata(cls) -> tuple[str, ...]:
        """Get all big data / distributed system schemes."""
        return (
            *Distributed.all,
            *Hadoop.all,
            *Lakehouse.all,
        )

    # @staticmethod
    @classgetter(cache=True)
    def cloud(cls) -> tuple[str, ...]:
        """Get all major cloud provider schemes (AWS, GCP, Azure storage)."""
        return (
            *AWSStorage.all,
            *AzureStorage.all,
            *GCPStorage.all,
        )

    # Class Getters END ----------------------------------------------------------------------------
