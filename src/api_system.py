import requests
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔹 1. Конфигурация эндпоинтов
API_ENDPOINTS = {
    "readings": {
        "path": "/api/v1/meter-readings",
        "method": "GET",
        "description": "Показания приборов учёта"
    },
    "charges": {
        "path": "/api/v1/billing/charges",
        "method": "GET",
        "description": "Начисления по лицевому счёту"
    }
}

class UtilityClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.timeout = timeout
        
        # Общая авторизация (Bearer, API-Key, Basic и т.д.)
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self.session.headers.update({"Accept": "application/json"})

    def _request(self, endpoint_key: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if endpoint_key not in API_ENDPOINTS:
            raise ValueError(f"Неизвестный тип запроса: {endpoint_key}")
            
        cfg = API_ENDPOINTS[endpoint_key]
        url = f"{self.base_url}{cfg['path']}"
        
        try:
            logger.info(f"Запрос: {cfg['description']} | URL: {url} | Params: {params}")
            resp = self.session.request(
                method=cfg["method"],
                url=url,
                params=params,          # Query-параметры (?key=value)
                timeout=self.timeout
            )
            resp.raise_for_status()     # Вызовет HTTPError при 4xx/5xx
            return resp.json()
            
        except requests.exceptions.HTTPError as e:
            # Часто API возвращает детали ошибки в JSON
            try:
                error_detail = e.response.json()
            except ValueError:
                error_detail = e.response.text
            logger.error(f"HTTP {e.response.status_code}: {error_detail}")
            raise RuntimeError(f"Ошибка API: {e.response.status_code} | {error_detail}") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"Сетевая ошибка: {e}")
            raise

    # 🔹 2.методы для бизнес-сущностей
    def get_readings(self, account_id: str, period: Optional[str] = None, resource: Optional[str] = None) -> Dict:
        params = {"account_id": account_id}
        if period:  params["period"] = period
        if resource: params["resource"] = resource  # вода, электричество и т.д.
        return self._request("readings", params=params)

    def get_charges(self, account_id: str, period: Optional[str] = None, billing_month: Optional[str] = None) -> Dict:
        params = {"account_id": account_id}
        if period:  params["period"] = period
        if billing_month: params["billing_month"] = billing_month
        return self._request("charges", params=params)

    def close(self):
        self.session.close()
    