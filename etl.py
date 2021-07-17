import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, dayofweek


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Loads song_data from S3 and processes it by extracting the songs and artist tables
        and then again loaded back to S3
    Args:
        spark: Spark Session.
        input_data: location of song_data json files with the songs metadata
        output_data: S3 bucket were dimensional tables in parquet format will be stored
    """
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select("song_id", "title", "artist_id", "year", "duration").dropDuplicates(['song_id'])
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy("year","artist_id").mode("overwrite").parquet(f"{output_data}/songs/songs_table")

    # extract columns to create artists table
    artists_table = df.selectExpr("artist_id", "artist_name as name", "artist_location as location", \
                                  "artist_latitude as latitude", "artist_longitude as longitude").dropDuplicates(['artist_id'])
    
    # write artists table to parquet files
    artists_table.write.mode("overwrite").parquet(f"{output_data}/artists/artists_table")

def process_log_data(spark, input_data, output_data):
    """
    Loads log_data from S3 and processes it by extracting the users, time, and songplay tables
        and then again loaded back to S3
    Args:
        spark: Spark Session.
        input_data: location of song_data json files with the songs metadata
        output_data: S3 bucket were dimensional tables in parquet format will be stored
    """
    
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data/*/*/*.json')
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')
    
    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter(df.page == "NextSong")

    # extract columns for users table    
    users_table = df.selectExpr("userId as user_id", "firstName as first_name", "lastName as last_name", "gender", "level").dropDuplicates(['userId'])
    
    # write users table to parquet files
    users_table.write.mode("overwrite").parquet(f"{output_data}/users/users_table")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: int((int(x)/1000)))
    df = df.withColumn('timestamp', get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x))
    df = df.withColumn('datetime', get_datetime(df.timestamp))
    
    # extract columns to create time table
    time_table = df.select(
        col('datetime').alias('start_time'),
        hour("datetime").alias('hour'), 
        dayofmonth("datetime").alias('day'), 
        weekofyear("datetime").alias('week'), 
        month("datetime").alias('month'), 
        year("datetime").alias('year'), 
        dayofweek("datetime").alias('weekday')
    )
    
    time_table = time_table.dropDuplicates(['start_time'])
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year","month").mode("overwrite").parquet(f"{output_data}/time/time_table")

    # read in song data to use for songplays table
    song_df = spark.read.json(song_data)

    # extract columns from joined song and log datasets to create songplays table 
    cond = [song_df.title == df.song, song_df.artist_name == df.artist, song_df.duration == df.length]
    
    songplays_table =  df.join(song_df, cond, 'left_outer')\
        .select(
            df.timestamp.alias('start_time'),
            col("userId").alias('user_id'),
            df.level,
            song_df.song_id,
            song_df.artist_id,
            col("sessionId").alias("session_id"),
            df.location,
            col("useragent").alias("user_agent"),
            year('datetime').alias('year'),
            month('datetime').alias('month'))
    
    songplays_table.select(monotonically_increasing_id().alias('songplay_id')).collect()
    
    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year", "month").mode('overwrite').parquet(f"{output_data}/songplays/songplays_table")

def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://udacity-course-bucket-tati"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
