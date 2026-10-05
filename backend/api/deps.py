from fastapi import Request

from backend.container import Services


def get_services(request: Request) -> Services:
    return request.app.state.services
