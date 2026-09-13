#!/usr/bin/env python3
import re,sys
p=sys.argv[1]
text=open(p,encoding='utf-8',errors='ignore').read().splitlines()
for ln in text:
    if '[CoordinateControllerAudit]' in ln or '[EpochSummary]' in ln or '[GeometryAuthorityAlert]' in ln or '[SampleGeometryOutlier]' in ln:
        print(ln)
