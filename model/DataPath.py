from enum import Enum, auto

DataPath={
    "WIFI" : "/wifi/wifi_weekly_",
    "SCHOOL" : "/Copenhagen_school/school_daily_",
    "SAFEGRAPH" : "/safegraph/sg_weekly_d10_",
    "WORKPLACE" : "/copresence-1/cop_1_hourly_",
    "LYONSCHOOL" : "/copresence-2/cop_2_hourly_",
    "HIGHSCHOOL" : "/copresence-3/cop_3_hourly_",
    "CONFERENCE" : "/sfhh/sfhh_cop_hourly_",
    "SFHH" : "/sfhh/sfhh_hourly_"}

ResultPath={
    "WIFI" : "wifi",
    "SCHOOL" : "school",
    "SAFEGRAPH" : "sg",
    "WORKPLACE" : "cop1",
    "LYONSCHOOL" : "cop2",
    "HIGHSCHOOL" : "cop3",
    "CONFERENCE" : "sfhh",
    "SFHH" : "sfhh"
    }