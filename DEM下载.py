import ee
import json

MY_PROJECT_ID = 'astute-diode-451504-e3'
try:
    ee.Initialize(project=MY_PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=MY_PROJECT_ID)
print("✅ [GEE] 初始化成功！")

with open('../data/shp/合肥市.geojson', 'r', encoding='utf-8') as f:
    hefei_json = json.load(f)

features = []
for feature in hefei_json['features']:
    geom = ee.Geometry(feature['geometry'])
    features.append(ee.Feature(geom))
hefei_fc = ee.FeatureCollection(features)
hefei_geometry = hefei_fc.geometry()

srtm = ee.Image('USGS/SRTMGL1_003')

hefei_dem = srtm.clip(hefei_geometry)

task = ee.batch.Export.image.toDrive(
    image=hefei_dem,
    description='Hefei_DEM_30m',
    folder='EarthEngine_Exports',
    fileNamePrefix='hefei_dem_srtm30',
    region=hefei_geometry,
    scale=30,  # 30米分辨率
    crs='EPSG:4326',
    maxPixels=1e13
)

task.start()
print("🚀 DEM 裁剪导出任务已提交到 GEE 云端！请去 Google Drive 查看。")