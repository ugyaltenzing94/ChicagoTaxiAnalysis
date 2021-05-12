/*Last updated 02/22/2021*/

/*process
data clean up (removing anomalies) and addition of more features
addition of daily weather data
addition of crime information for all Chicago Community Areas
addition of Chicago Sides and Community Area mapping information
Undersampling the dataset to create balanced distribution of Sides for model training
*/

/*modeling data
two training datasets created: one with data from 2016-19 and the other from 2016-20
one test dataset created containing data from Jan 2020 and Jan 2021
*/



--0--
----------------------------LOADING MAIN DATA INTO OUR PROJECT----------------------------

CREATE OR REPLACE TABLE
  `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps` AS
SELECT
  *
FROM
  `bigquery-public-data.chicago_taxi_trips.taxi_trips`
  
  
--1--
----------------------------CLEAN TAXI DATASET WITH SPECIAL DAYS AND OTHER FILTERS----------------------------

CREATE OR REPLACE TABLE
  `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_cln` AS(
  SELECT
    trip_start_timestamp,
    pickup_community_area,
    trip_total,
    fare,
    extras,
    tips,
    tolls,
    company,
    payment_type,
    trip_seconds,
    trip_miles,
    --extracting hour of the day
    EXTRACT (HOUR
    FROM
      trip_start_timestamp) AS hour,
    --extracting day of the week
    EXTRACT (DAYOFWEEK
    FROM
      trip_start_timestamp) AS day_of_week,
    --extracting date
    EXTRACT (Date
    FROM
      trip_start_timestamp) AS date,
    --extracting day of the week
    EXTRACT (Day
    FROM
      trip_start_timestamp) AS day,
    --extracting year
    EXTRACT(YEAR
    FROM
      trip_start_timestamp) AS year,
    --extracting month of the year
    EXTRACT (MONTH
    FROM
      trip_start_timestamp) AS month,
    --extracting week of the year
    EXTRACT(WEEK
    FROM
      trip_start_timestamp) AS week,
  IF
    (EXTRACT(DAYOFWEEK
      FROM
        trip_start_timestamp) = 7
      OR EXTRACT(DAYOFWEEK
      FROM
        trip_start_timestamp)=1,
      1,
      0) AS weekend,
  IF
    (
      --Labour Day
      EXTRACT(DAYOFWEEK
      FROM
        trip_start_timestamp) = 2
      AND EXTRACT (MONTH
      FROM
        trip_start_timestamp) = 9
      AND CEILING(EXTRACT (Day
        FROM
          trip_start_timestamp)/7)=1
      --Thanksgiving
      OR EXTRACT (MONTH
      FROM
        trip_start_timestamp) = 11
      AND EXTRACT(DAYOFWEEK
      FROM
        trip_start_timestamp) = 5
      AND CEILING(EXTRACT (Day
        FROM
          trip_start_timestamp)/7)= 4
      --Christmas
      OR EXTRACT (MONTH
      FROM
        trip_start_timestamp) = 12
      AND EXTRACT (Day
      FROM
        trip_start_timestamp)/7= 25
      --Independence Day
      OR EXTRACT (MONTH
      FROM
        trip_start_timestamp) = 5
      AND EXTRACT (Day
      FROM
        trip_start_timestamp)= 4
      --St. Patrick's Day
      OR EXTRACT (MONTH
      FROM
        trip_start_timestamp) = 3
      AND EXTRACT (Day
      FROM
        trip_start_timestamp) = 17
      --New Year's Eve
      OR EXTRACT (MONTH
      FROM
        trip_start_timestamp) = 12
      AND EXTRACT (Day
      FROM
        trip_start_timestamp) = 31
      --Martin Luther King Day
      OR EXTRACT (MONTH
      FROM
        trip_start_timestamp) = 1
      AND EXTRACT(DAYOFWEEK
      FROM
        trip_start_timestamp) = 2
      AND CEILING(EXTRACT (Day
        FROM
          trip_start_timestamp)/7)= 3
      --Valentine's Day
      OR EXTRACT (MONTH
      FROM
        trip_start_timestamp) = 2
      AND EXTRACT (Day
      FROM
        trip_start_timestamp) = 14,
      1,
      0) AS special_days
  FROM (
    SELECT
      *,
      trip_miles/(trip_seconds/3600) AS trip_speed
    FROM
      `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps`
    WHERE
      --Removing rows based on highest number of null values in columns 'company' and 'pickup_community_area'
      company IS NOT NULL
      AND pickup_community_area IS NOT NULL
      --Adding reasonable filters for trip time, distance and fare
      AND trip_seconds>=60
      AND trip_miles>0
      AND trip_total>=1
      AND (fare<=1000
        AND extras<=1000
        AND tolls<=1000
        AND tips<=1000
        AND trip_total<=1000)
      --Extracting data from 2016 onwards for our model
      AND EXTRACT(Year
      FROM
        trip_start_timestamp)>=2016
      --ratio of trip total and trip miles should be greater than two
      AND (trip_total/trip_miles >=2
        --to prevent high outliers the same ratio should be less than 10
        AND trip_total/trip_miles<=10) )
  WHERE
    --With the city traffic and speed restrictions, speed was kept less than 70
    trip_speed>=1
    AND trip_speed<=70 )
  
  
 --2--
----------------------------CHICAGO DAILY WEATHER DATA----------------------------
--Daily summaries of precipitation, snow and temperature for Chicago were downloaded from 'https://www.ncdc.noaa.gov/cdo-web/search'
--and the csv was uploaded to `us-gcp-ame-con-01e-npd-1.chcg_dtst.weather_daily`, calculated averages for each datapoint over all the Chicago stations
CREATE OR REPLACE TABLE
  `us-gcp-ame-con-01e-npd-1.chcg_dtst.weather_daily_averages` AS (
  SELECT
    DATE,
    ROUND(AVG(prcp),2) AS avg_prcp,
    ROUND(AVG(snow),2) AS avg_snow,
    ROUND(AVG(CAST(tavg AS FLOAT64)),2) AS avg_tavg
  FROM
    `us-gcp-ame-con-01e-npd-1.chcg_dtst.weather_daily`
  WHERE
    EXTRACT(Year
    FROM
      DATE) >2015
  GROUP BY
    DATE
  ORDER BY
    DATE )
  
  
--3--
----------------------------CHICAGO CRIME DATA----------------------------
--Added crime rating for each Chicago Community Area by normalizing total crime count over the last 10 years
CREATE OR REPLACE TABLE
  `us-gcp-ame-con-01e-npd-1.chcg_dtst.chicago_crime` AS (
  WITH
    crime_data AS (
    SELECT
      COUNT(unique_key) AS crime_count,
      community_area
    FROM
      `bigquery-public-data.chicago_crime.crime`
    WHERE
      year > 2010
      AND community_area IS NOT NULL
    GROUP BY
      community_area )
  SELECT
    (crime_count - (
      SELECT
        MIN(crime_count)
      FROM
        crime_data))/((
      SELECT
        MAX(crime_count)
      FROM
        crime_data) - (
      SELECT
        MIN(crime_count)
      FROM
        crime_data)) AS crime_rating,
    community_area,
    crime_count
  FROM
    crime_data
  GROUP BY
    community_area,
    crime_count
  ORDER BY
    crime_rating DESC)
  
  
--4--
----------------------------TABLE JOINS FOR COLLECTIVE EDA----------------------------
--picking data from 2016-2019 for EDA and model training and evaluation
CREATE OR REPLACE TABLE
  `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda` AS(
  SELECT
    * EXCEPT (Date,
      crime_count,
      Community_Area_Number,
      Name,
      community_area)
  FROM (
    SELECT
      *
    FROM
      `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_cln`
    WHERE
      EXTRACT(Year
      FROM
        trip_start_timestamp) < 2020) AS taxi_trip_dataset
  JOIN
    `us-gcp-ame-con-01e-npd-1.chcg_dtst.comm_area` AS community_area_dataset
  ON
    taxi_trip_dataset.pickup_community_area = community_area_dataset.Community_Area_Number
  JOIN
    `us-gcp-ame-con-01e-npd-1.chcg_dtst.chicago_crime` AS crime_data_set
  ON
    taxi_trip_dataset.pickup_community_area = crime_data_set.community_area
  JOIN
    `us-gcp-ame-con-01e-npd-1.chcg_dtst.weather_daily_averages` AS weather_data_set
  ON
    taxi_trip_dataset.Date = weather_data_set.Date )
	
--picking data from 2016-2020 for EDA and model training and evaluation
CREATE OR REPLACE TABLE
  `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda2` AS(
  SELECT
    * EXCEPT (Date,
      crime_count,
      Community_Area_Number,
      Name,
      community_area)
  FROM (
    SELECT
      *
    FROM
      `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_cln`
    WHERE
      EXTRACT(Year
      FROM
        trip_start_timestamp) <= 2020) AS taxi_trip_dataset
  JOIN
    `us-gcp-ame-con-01e-npd-1.chcg_dtst.comm_area` AS community_area_dataset
  ON
    taxi_trip_dataset.pickup_community_area = community_area_dataset.Community_Area_Number
  JOIN
    `us-gcp-ame-con-01e-npd-1.chcg_dtst.chicago_crime` AS crime_data_set
  ON
    taxi_trip_dataset.pickup_community_area = crime_data_set.community_area
  JOIN
    `us-gcp-ame-con-01e-npd-1.chcg_dtst.weather_daily_averages` AS weather_data_set
  ON
    taxi_trip_dataset.Date = weather_data_set.Date )
	

--5--
----------------------------CREATING UNDERSAMPLED DATA----------------------------
--Unbalanced class distribution of Chicago Sides balanced as following 
--based on the lowest number of records for 'Far Southwest Side' in the `tx_trps_eda` and `tx_trps_eda2` table.
--This is the final dataset used for model training
--2016-19 data taken from 'tx_trps_eda`
CREATE OR REPLACE TABLE
  `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_undrsmpld` AS (
  SELECT
    UNIX_SECONDS(trip_start_timestamp) AS trip_start_timestamp,
    payment_type,
    hour,
    day_of_week,
    month,
    weekend,
    special_days,
    trip_total,
    trip_miles,
    crime_rating,
    avg_prcp,
    avg_snow,
    avg_tavg,
    side_number side
  FROM
    `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda`
  WHERE
    (RAND() < 22000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda`
      WHERE
        side = 'Central'
        AND trip_start_timestamp < '2019-12-31' )
      AND side = 'Central'
      AND trip_start_timestamp < '2019-12-31')
    OR (RAND() < 22000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda`
      WHERE
        side = 'South Side'
        AND trip_start_timestamp < '2019-12-31' )
      AND side = 'South Side'
      AND trip_start_timestamp < '2019-12-31')
    OR (RAND() < 22000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda`
      WHERE
        side = 'West Side'
        AND trip_start_timestamp < '2019-12-31')
      AND side = 'West Side'
      AND trip_start_timestamp < '2019-12-31')
    OR (RAND() < 22000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda`
      WHERE
        side = 'Far North Side'
        AND trip_start_timestamp < '2019-12-31')
      AND side = 'Far North Side'AND trip_start_timestamp < '2019-12-31')
    OR (RAND() < 22000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda`
      WHERE
        side = 'Southwest Side'
        AND trip_start_timestamp < '2019-12-31' )
      AND side = 'Southwest Side'
      AND trip_start_timestamp < '2019-12-31')
    OR (RAND() < 22000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda`
      WHERE
        side = 'Northwest Side'
        AND trip_start_timestamp < '2019-12-31' )
      AND side = 'Northwest Side'
      AND trip_start_timestamp < '2019-12-31')
    OR (RAND() < 22000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda`
      WHERE
        side = 'Far Southeast Side'
        AND trip_start_timestamp < '2019-12-31' )
      AND side = 'Far Southeast Side'
      AND trip_start_timestamp < '2019-12-31')
    OR (RAND() < 22000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda`
      WHERE
        side = 'Far Southwest Side'
        AND trip_start_timestamp < '2019-12-31' )
      AND side = 'Far Southwest Side'
      AND trip_start_timestamp < '2019-12-31')
    OR (RAND() < 22000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda`
      WHERE
        side = 'North Side'
        AND trip_start_timestamp < '2019-12-31')
      AND side = 'North Side'
      AND trip_start_timestamp < '2019-12-31') )

--2016-20 data taken from 'tx_trps_eda2`
CREATE OR REPLACE TABLE
  `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_undrsmpld2` AS (
  SELECT
    UNIX_SECONDS(trip_start_timestamp) AS trip_start_timestamp,
    payment_type,
    hour,
    day_of_week,
    month,
    weekend,
    special_days,
    trip_total,
    trip_miles,
    crime_rating,
    avg_prcp,
    avg_snow,
    avg_tavg,
    side_number side
  FROM
    `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda2`
  WHERE
    (RAND() < 41000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda2`
      WHERE
        side = 'Central'
        AND trip_start_timestamp < '2020-12-31' )
      AND side = 'Central'
      AND trip_start_timestamp < '2020-12-31')
    OR (RAND() < 41000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda2`
      WHERE
        side = 'South Side'
        AND trip_start_timestamp < '2020-12-31' )
      AND side = 'South Side'
      AND trip_start_timestamp < '2020-12-31')
    OR (RAND() < 41000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda2`
      WHERE
        side = 'West Side'
        AND trip_start_timestamp < '2020-12-31')
      AND side = 'West Side'
      AND trip_start_timestamp < '2020-12-31')
    OR (RAND() < 41000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda2`
      WHERE
        side = 'Far North Side'
        AND trip_start_timestamp < '2020-12-31')
      AND side = 'Far North Side'AND trip_start_timestamp < '2020-12-31')
    OR (RAND() < 41000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda2`
      WHERE
        side = 'Southwest Side'
        AND trip_start_timestamp < '2020-12-31' )
      AND side = 'Southwest Side'
      AND trip_start_timestamp < '2020-12-31')
    OR (RAND() < 41000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda2`
      WHERE
        side = 'Northwest Side'
        AND trip_start_timestamp < '2020-12-31' )
      AND side = 'Northwest Side'
      AND trip_start_timestamp < '2020-12-31')
    OR (RAND() < 41000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda2`
      WHERE
        side = 'Far Southeast Side'
        AND trip_start_timestamp < '2020-12-31' )
      AND side = 'Far Southeast Side'
      AND trip_start_timestamp < '2020-12-31')
    OR (RAND() < 41000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda2`
      WHERE
        side = 'Far Southwest Side'
        AND trip_start_timestamp < '2020-12-31' )
      AND side = 'Far Southwest Side'
      AND trip_start_timestamp < '2020-12-31')
    OR (RAND() < 41000/(
      SELECT
        COUNT(*)
      FROM
        `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda2`
      WHERE
        side = 'North Side'
        AND trip_start_timestamp < '2020-12-31')
      AND side = 'North Side'
      AND trip_start_timestamp < '2020-12-31') )
	
--6--
----------------------------TABLE JOINS FOR TEST DATA----------------------------
--picking data from Jan 2020 and Jan 2021 for model testing
CREATE OR REPLACE TABLE
  `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_tst` AS(
  SELECT
    UNIX_SECONDS(trip_start_timestamp) AS trip_start_timestamp,
    payment_type,
    hour,
    day_of_week,
    month,
    weekend,
    special_days,
    trip_total,
    trip_miles,
    crime_rating,
    avg_prcp,
    avg_snow,
    avg_tavg,
    side
  FROM (
    SELECT
      *
    FROM
      `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_cln`
    WHERE
      (EXTRACT(Year
        FROM
          trip_start_timestamp) = 2020
        OR EXTRACT(Year
        FROM
          trip_start_timestamp) = 2021)
      AND (EXTRACT (MONTH
        FROM
          trip_start_timestamp) = 1)) AS taxi_trip_dataset
  JOIN
    `us-gcp-ame-con-01e-npd-1.chcg_dtst.comm_area` AS community_area_dataset
  ON
    taxi_trip_dataset.pickup_community_area = community_area_dataset.Community_Area_Number
  JOIN
    `us-gcp-ame-con-01e-npd-1.chcg_dtst.chicago_crime` AS crime_data_set
  ON
    taxi_trip_dataset.pickup_community_area = crime_data_set.community_area
  JOIN
    `us-gcp-ame-con-01e-npd-1.chcg_dtst.weather_daily_averages` AS weather_data_set
  ON
    taxi_trip_dataset.Date = weather_data_set.Date )