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

    class schemes:
        s3: tuple[str, ...] = ("s3", "s3a", "s3n")


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

    class schemes:
        abfs: tuple[str, ...] = ("abfs", "abfss")
        wasb: tuple[str, ...] = ("wasb", "wasbs")


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

    class schemes:
        neo4j: tuple[str, ...] = ("neo4j", "neo4js")


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

    class schemes:
        huggingface: tuple[str, ...] = ("hf", "huggingface")


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

    class schemes:
        smb: tuple[str, ...] = ("smb", "cifs")


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

    class schemes:
        cassandra: tuple[str, ...] = ("cassandra", "cql")
        mongo: tuple[str, ...] = ("mongo", "mongodb")
        redis: tuple[str, ...] = ("redis", "rediss")


class Search(SchemeBase):
    """Search and vector database URI schemes."""

    elasticsearch = "elasticsearch"
    es = "es"  # Elasticsearch alias
    meilisearch = "meilisearch"
    opensearch = "opensearch"
    solr = "solr"
    typesense = "typesense"

    class schemes:
        elasticsearch: tuple[str, ...] = ("elasticsearch", "es")


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

    class schemes:
        cockroach: tuple[str, ...] = ("cockroach", "cockroachdb")
        mssql: tuple[str, ...] = ("mssql", "sqlserver")
        postgres: tuple[str, ...] = ("postgres", "postgresql")


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

    class schemes:
        chroma: tuple[str, ...] = ("chroma", "chromadb")


class Web(SchemeBase):
    """Web protocol URI schemes."""

    ftp = "ftp"
    ftps = "ftps"
    http = "http"
    https = "https"

    class schemes:
        ftp: tuple[str, ...] = ("ftp", "ftps")
        http: tuple[str, ...] = ("http", "https")


class Schemes:
    """
    Collection access for URI schemes, always returns tuple[str, ...].

            **Cloud Storage:**
                - `Schemes.aws.storage` - AWS S3 storage schemes (s3, s3a, s3n)
                - `Schemes.azure.storage` - Azure storage schemes
                - `Schemes.cloud` - All cloud provider storage schemes (AWS + Azure + GCP)
                - `Schemes.gcp.storage` - GCP storage schemes

            **Databases:**
                - `Schemes.db.all` - All database schemes
                - `Schemes.db.cloud` - Cloud-managed databases (AWS + Azure + GCP)
                - `Schemes.aws.database` - AWS databases (redshift, dynamodb, athena, etc.)
                - `Schemes.gcp.database` - GCP databases (bigquery, bigtable, spanner, etc.)
                - `Schemes.azure.database` - Azure databases (cosmosdb, synapse, etc.)
                - `Schemes.db.nosql` - NoSQL databases (mongodb, redis, cassandra, etc.)
                - `Schemes.db.vector` - Vector databases (pinecone, weaviate, qdrant, etc.)
                - `Schemes.db.search` - Search databases (elasticsearch, opensearch, etc.)
                - `Schemes.db.timeseries` - Time series databases (influxdb, prometheus, etc.)
                - `Schemes.db.graph` - Graph databases (neo4j, arangodb, etc.)
                - `Schemes.db.analytical` - Analytical databases (clickhouse, snowflake, etc.)
                - `Schemes.db.sql` - SQL databases (sqlite, mysql, postgresql)

            **Distributed Systems:**
                - `Schemes.distributed` - All distributed data platform schemes
                  (Distributed FS, Hadoop, Lakehouse: hdfs, alluxio, delta, iceberg, etc.)

            **Local:**
                - `Local.all` - local and URN schemes (file, urn)

            **ML Platforms:**
                - `Schemes.ml.all` - All ML-related schemes
                - `Schemes.ml.mlflow` - MLflow (runs, models)
                - `Schemes.ml.tracking` - Experiment tracking (wandb, comet, neptune, clearml)
                - `Schemes.ml.hub` - Model hubs (hf, torchhub, tfhub, onnx)
                - `Schemes.ml.data_versioning` - Data versioning (dvc, pachyderm)
                - `Schemes.ml.datasets` - Dataset schemes (tfds, torch)

            **Web:**
                - `Web.all` - web related schemes (http, https, ftp, ftps)

    Examples:
        >>> Schemes.aws.storage
        ('s3', 's3a', 's3n')

        >>> Schemes.db.vector
        ('chroma', 'chromadb', 'milvus', 'pinecone', 'qdrant', 'weaviate')

        >>> Schemes.ml.tracking
        ('aim', 'clearml', 'comet', 'mlflow', 'neptune', 'sacred', 'tensorboard', 'wandb')

        >>> 'wandb' in Schemes.ml.all and 'hf' in Schemes.ml.all
        True

        >>> 'bigquery' in Schemes.db.all and 'pinecone' in Schemes.db.all
        True

        >>> 'hdfs' in Schemes.distributed and 'delta' in Schemes.distributed
        True
    """

    # ── Cloud providers ───────────────────────────────────────────────────────

    class aws:
        """All AWS URI scheme collections."""

        storage: tuple[str, ...] = AWSStorage.all
        database: tuple[str, ...] = AWSDatabase.all

        @classgetter(cache=True)
        def all(cls) -> tuple[str, ...]:
            """All AWS schemes (storage + database)."""
            return (*AWSStorage.all, *AWSDatabase.all)

    class azure:
        """All Azure URI scheme collections."""

        storage: tuple[str, ...] = AzureStorage.all
        database: tuple[str, ...] = AzureDatabase.all

        @classgetter(cache=True)
        def all(cls) -> tuple[str, ...]:
            """All Azure schemes (storage + database)."""
            return (*AzureStorage.all, *AzureDatabase.all)

    class gcp:
        """All GCP URI scheme collections."""

        storage: tuple[str, ...] = GCPStorage.all
        database: tuple[str, ...] = GCPDatabase.all

        @classgetter(cache=True)
        def all(cls) -> tuple[str, ...]:
            """All GCP schemes (storage + database)."""
            return (*GCPStorage.all, *GCPDatabase.all)

    # ── Databases ─────────────────────────────────────────────────────────────

    class db:
        """All database URI scheme collections."""

        sql: tuple[str, ...] = SQL.all
        nosql: tuple[str, ...] = NoSQL.all
        vector: tuple[str, ...] = Vector.all
        graph: tuple[str, ...] = Graph.all
        analytical: tuple[str, ...] = Analytical.all
        timeseries: tuple[str, ...] = TimeSeries.all
        search: tuple[str, ...] = Search.all
        cloud: tuple[str, ...] = (*AWSDatabase.all, *AzureDatabase.all, *GCPDatabase.all)

        @classgetter(cache=True)
        def all(cls) -> tuple[str, ...]:
            """All database schemes across all types and cloud providers."""
            return (
                *SQL.all,
                *NoSQL.all,
                *Vector.all,
                *Graph.all,
                *Analytical.all,
                *TimeSeries.all,
                *Search.all,
                *AWSDatabase.all,
                *AzureDatabase.all,
                *GCPDatabase.all,
            )

    # ── ML/AI ─────────────────────────────────────────────────────────────────

    class ml:
        """All ML/AI platform URI scheme collections."""

        tracking: tuple[str, ...] = MLTracking.all
        hub: tuple[str, ...] = MLHub.all
        datasets: tuple[str, ...] = MLDataset.all
        data_versioning: tuple[str, ...] = DataVersioning.all
        mlflow: tuple[str, ...] = MLFlow.all

        @classgetter(cache=True)
        def all(cls) -> tuple[str, ...]:
            """All ML/AI schemes across tracking, hubs, datasets, and versioning."""
            return (
                *MLTracking.all,
                *MLHub.all,
                *MLDataset.all,
                *DataVersioning.all,
                *MLFlow.all,
            )

    # ── Cross-cutting classgetters ────────────────────────────────────────────

    @classgetter(cache=True)
    def all(cls) -> tuple[str, ...]:
        """All known URI schemes."""
        return (
            *AWSStorage.all,
            *AWSDatabase.all,
            *AzureStorage.all,
            *AzureDatabase.all,
            *GCPStorage.all,
            *GCPDatabase.all,
            *SQL.all,
            *NoSQL.all,
            *Vector.all,
            *Graph.all,
            *Analytical.all,
            *TimeSeries.all,
            *Search.all,
            *MLTracking.all,
            *MLHub.all,
            *MLDataset.all,
            *DataVersioning.all,
            *MLFlow.all,
            *Distributed.all,
            *Hadoop.all,
            *Lakehouse.all,
            *NetworkFS.all,
            *Local.all,
            *Web.all,
        )

    @classgetter(cache=True)
    def cloud(cls) -> tuple[str, ...]:
        """All cloud provider storage schemes (AWS + Azure + GCP)."""
        return (*AWSStorage.all, *AzureStorage.all, *GCPStorage.all)

    @classgetter(cache=True)
    def distributed(cls) -> tuple[str, ...]:
        """All distributed data platform schemes (distributed FS, Hadoop, lakehouse)."""
        return (*Distributed.all, *Hadoop.all, *Lakehouse.all)
