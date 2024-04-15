import astropy.utils.data as aud

print("Current cached urls:")
print(aud.get_cached_urls())
print("Saving old cache to old_cache.zip")
aud.export_download_cache("old_cache.zip")
aud.clear_download_cache()
print("Cleared cached urls:")
print(aud.get_cached_urls())
print("Importing from cache.zip")
aud.import_download_cache("cache.zip")
print("New cached urls:")
print(aud.get_cached_urls())
