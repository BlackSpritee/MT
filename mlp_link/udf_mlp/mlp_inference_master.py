#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from pyspark.sql.types import *

def output_schema(schema ,params):
    fields = [StructField("reviewid", StringType()),
              StructField("sub", StringType()),
              StructField("obj", StringType()),
              StructField("sub_type", StringType()),
              StructField("obj_type", StringType()),
              StructField("reviewbody", StringType()),
              StructField("predict_rel", StringType()),
              StructField("predict_score", FloatType()),
              ]
    schema = StructType(fields)
    return schema
