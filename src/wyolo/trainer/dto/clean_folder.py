from pydantic import BaseModel


class CleanFolderObj(BaseModel):
    """
    DTO for cleaning a folder.
    """

    project_path: str
