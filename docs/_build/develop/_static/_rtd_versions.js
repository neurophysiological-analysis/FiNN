window.addEventListener("DOMContentLoaded", (event) => {
    let VersionsElement = document.getElementsByClassName("rst-versions")
    let CurrentVersionsElement = document.getElementsByClassName("rst-current-version")

    Array.from(CurrentVersionsElement).forEach(element => {
        element.addEventListener("click", (e) => {
            Array.from(VersionsElement).forEach(versionselement => {
                versionselement.classList.toggle("shift-up")
            })
        })
    })
});
