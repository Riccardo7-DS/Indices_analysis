
from utils.function_clns import config
import os
import xarray as xr
from typing import Literal

class EeModis():
    def __init__(self,
                 start_date:str, 
                 end_date:str,
                 name:str=Literal["ref_061", "NDVI_06"], 
                 output_resolution:int=1000,
                 download_collection:bool=True):
        
        valid_names = {"ref_061", "NDVI_06"}
        assert name in valid_names, \
            "Invalid value for 'name'. It should be one of: 'ref_061', 'NDVI_06'"

        import ee
        import geemap

        ee.Initialize()
        self.start_date = start_date
        self.end_date = end_date
        self.out_dir = config["MODIS"][name]["output_dir"]
        self.polygon = self.ee_hoa_geometry()
        self.output_resolution = output_resolution
        self.product = self._get_product_name(name)
    
        # Import images
        images = ee.ImageCollection(self.product)\
                    .filterDate(start_date, end_date)\
                    .filterBounds(self.polygon)
        
        bands = self._get_bands(self.product)
        img_bands = images.select(bands)

        if name == "ref_061":
            img_bands = img_bands.map(lambda x: self._compute_ndvi(x, bands[1], bands[0],
                                                                   bands[2]))
        
        if download_collection is True:
            self._collection_prepr_download(img_bands)
        else:
            self._image_prepr_download(img_bands)

    def _extract_qa(self, img, qaBand):
        qa_mask = img.select(qaBand)
        quality_mask = self._bitwise_extract(qa_mask, 0, 1).eq(0)
        return quality_mask

    def _bitwise_extract(self, input_image, from_bit, to_bit):
        import ee
        mask_size = ee.Number(1).add(to_bit).subtract(from_bit)
        mask = ee.Number(1).leftShift(mask_size).subtract(1)
        return input_image.rightShift(from_bit).bitwiseAnd(mask)

    def _compute_ndvi(self, img, nirBand, redBand, qaBand=None):
        ndvi = img.normalizedDifference([nirBand, redBand])
        if qaBand is not None:
            qa = self._extract_qa(img, qaBand)
            return ndvi.updateMask(qa)
        else:
            return ndvi
    
    def _get_product_name(self, name):
        return config["MODIS"][name]["product"]

    def _get_bands(self, name):
        if name == "MODIS/MOD09GA_006_NDVI":
            return 'NDVI'
        elif name == "MODIS/061/MOD09GQ":
            return ["sur_refl_b01", "sur_refl_b02","QC_250m"]
        else:
            raise NotImplementedError(f"Product {name} not implemented")


    def _imreproj(self, image):
        return image.reproject(crs='EPSG:4326', scale=self.output_resolution)

    def ee_hoa_geometry(self):
        import ee
        polygon = ee.Geometry.Polygon(
            [[[30.288233396779802,-5.949173816626356],
            [51.9972177717798,-5.949173816626356],
            [51.9972177717798,15.808293611760663],
            [30.288233396779802,15.808293611760663]]]
        )
        return  ee.Geometry(polygon, None, False)
        
    def _collection_prepr_download(self, images):
        import geemap
        clipped_img = images.map(lambda image: image.clip(self.polygon))
        reprojected_img = clipped_img.map(self._imreproj)
        reprojected_img.aggregate_array("system:index").getInfo()
        geemap.ee_export_image_collection(reprojected_img, self.out_dir)        


    def _image_prepr_download(self, images):
        import ee
        image_ids = images.aggregate_array("system:index").getInfo()
            # Download each image
        for idx, image_id in enumerate(image_ids):
            image = ee.Image(image_id).clip(self.polygon)\
            .reproject('EPSG:4326', None, self.output_resolution)

            # Get the original CRS and geotransform of the image
            proj = image.projection().getInfo()

            # Create a filename for the downloaded image
            filename = image_id.split("/")[-1]

            # Export the image with the original CRS and geotransform
            task = ee.batch.Export.image.toDrive(
                image = image,
                region = self.polygon.bounds(), # Or use custom ee.Geometry.Rectangle([minlon, minlat, maxlon, maxlat])
                description = filename,
                folder = "MODIS_NDVI",
                crs = proj["crs"],
                crsTransform = proj["transform"],
                maxPixels = 1e13,
                fileFormat = "GeoTIFF"
            )
            task.start()
            print(f"Exporting {filename}...")
    
    def _preprocess_file(self, ds:xr.Dataset):
        import pandas as pd
        time = ds.encoding['source'].split("/")[-1].replace("_","/")[:-4]
        date_xr = pd.to_datetime(time)
        ds = ds.assign_coords(time=date_xr)
        ds = ds.expand_dims(dim="time")
        return ds.isel(band=0)

    def xarray_preprocess(self):
        import xarray as xr
        import os
        files = [os.path.join(self.out_dir,f) for f in os.listdir(self.out_dir) if f.endswith(".tif")]
        dataset = xr.open_mfdataset(files, preprocess=self._preprocess_file, engine="rasterio")
        return dataset


def earthaccess_download(product_name,
                         targetdir, 
                         start_date="2000-01-01", 
                         end_date="2023-12-31"):
    import earthaccess

    earthaccess.login()

    results = earthaccess.search_data(
        short_name=product_name,
        cloud_hosted=True,
        polygon= [(30.288233396779802,-5.949173816626356),
                (51.9972177717798,-5.949173816626356),
                (51.9972177717798,15.808293611760663),
                (30.288233396779802,15.808293611760663),
                (30.288233396779802,-5.949173816626356),],
        temporal=(start_date, end_date)
        )
    files = earthaccess.download(results, targetdir)

def modistools_download(user:str, 
                        password:str, 
                        targetdir:str,
                        product_name:str,
                        start_date:str="2000-01-01", 
                        end_date:str="2023-12-31"):
    
    from modis_tools.auth import ModisSession
    from modis_tools.resources import CollectionApi, GranuleApi
    from modis_tools.granule_handler import GranuleHandler

    session = ModisSession(username=user, password=password)

    # Query the MODIS catalog for collections
    collection_client = CollectionApi(session=session)
    collections = collection_client.query(short_name=product_name)
    # Query the selected collection for granules
    granule_client = GranuleApi.from_collection(collections[0], session=session)

    # Filter the selected granules via spatial and temporal parameters
    bbox = [30.288233396779802,-5.949173816626356,  51.9972177717798,  15.808293611760663]
    granules = granule_client.query(start_date="2019-12-29", 
                                            end_date="2019-12-31", 
                                            bounding_box=bbox)

    # Download the granules
    GranuleHandler.download_from_granules(granules, 
                                          session, 
                                          threads=-1,
                                          path=targetdir)

