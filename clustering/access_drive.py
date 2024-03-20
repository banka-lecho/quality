from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/drive']

SERVICE_ACCOUNT_FILE = '/Users/anastasiaspileva/quality-assessment/clustering/client_secret_57755367914-ic06kv1s7h55s6vp372piaunpula20i6.apps.googleusercontent.com.json'

credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)

folder_id = 'clean'

results = service.files().list(q=f"mimeType='image/jpeg' and '{folder_id}' in parents",
                               spaces='drive',
                               fields='files(id, name)',
                               pageToken=None, supportsAllDrives=True).execute()

for file in results.get('files', []):
    print('Found file: %s (%s)' % (file.get('name'), file.get('id')))
