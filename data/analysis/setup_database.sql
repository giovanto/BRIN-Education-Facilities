
CREATE TABLE onderwijs_historical (
    id VARCHAR(10) PRIMARY KEY,
    name VARCHAR(200),
    geom GEOMETRY(Point, 4326),
    type VARCHAR(50),
    subtype VARCHAR(50),
    start_date DATE,
    end_date DATE,
    municipality VARCHAR(100),
    province VARCHAR(50),
    active_1950_1959 BOOLEAN,
    active_1960_1969 BOOLEAN,
    active_1970_1979 BOOLEAN,
    active_1980_1989 BOOLEAN,
    active_1990_1999 BOOLEAN,
    active_2000_2009 BOOLEAN,
    active_2010_2019 BOOLEAN,
    active_2020_2029 BOOLEAN
);

-- Create spatial index
CREATE INDEX education_locations_geom_idx ON education_locations USING GIST (geom);

-- Create indexes for temporal filtering
CREATE INDEX education_locations_temporal_idx ON education_locations USING btree (
    active_1950_1959, active_1960_1969, active_1970_1979, 
    active_1980_1989, active_1990_1999, active_2000_2009, 
    active_2010_2019, active_2020_2029
);

-- Create index for name search
CREATE INDEX education_locations_name_idx ON education_locations USING btree (name text_pattern_ops);

-- Import data
\COPY education_locations FROM 'education_locations.csv' WITH (FORMAT csv, HEADER true);
