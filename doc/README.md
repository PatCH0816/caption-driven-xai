# MAster thesis
Document is based on: [Template for writing a PhD thesis in Markdown](https://github.com/tompollard/phd_thesis_markdown)

# Setup

## Tex Live

See: [TeX Live - Manuell](https://wiki.ubuntuusers.de/TeX_Live/#Manuell)

Da die durch Ubuntu-/Debian-Pakete zur Verfügung gestellten Pakete den Latex-Paketen auf dem CTAN-Server häufig eine gewisse Zeit hinterherhinken, kann es sich für erfahrene Benutzer empfehlen, TeX Live manuell zu installieren. Dies gilt besonders für Nutzer, die XeTeX oder LuaTeX benutzen möchten oder sonstige aktuelle Pakete benötigen.

Dazu folgt hier eine Anleitung, die im Diskussionsforum zu dieser Seite entwickelt wurde. Obwohl einige Schritte über eine grafische Oberfläche ausgeführt werden könnten, wird hier nur das Vorgehen über das Terminal beschrieben:

* Zunächst muss sichergestellt werden, dass alle TeX Live-Pakete aus den offiziellen Paketquellen deinstalliert sind, man kann zum Beispiel mit Synaptic nach texlive suchen oder den folgenden Befehl dafür nutzen:
    ```
    apt search texlive | grep -i install
    ```
* Einen temporären Ordner für die Installationsdateien erstellen und hinein wechseln:
    ```
    mkdir install-tl && cd install-tl 
    ```
* Das aktuelle Installationsskript herunterladen und im aktuellen Ordner entpacken:
    ```
    wget -O - -- http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz | tar xzf - --strip-components=1
    ```
* Ggf. Terminalfenster maximieren, damit während der Installation alle Optionen sichtbar sind.
* Temporäre Rootrechte erlangen:
    ```
    sudo -s 
    ```
* Abhängige Pakete installieren:
    ```
    apt install tex-common texinfo equivs perl-tk perl-doc 
    ```
* Installation starten:
    ```
    ./install-tl 
    ```
* Menüpunkt „Options“ wählen: O ⏎
* Menüpunkt „create symlinks in standard directories“ wählen: L ⏎
* Die drei darauf folgenden Anfragen für Pfadänderungen mit Enter bestätigen (also die Vorgaben annehmen)
* Zurück ins Hauptmenu: R ⏎
* Menüpunkt „set installation scheme“ wählen: S ⏎
* Menüpunkt „medium scheme (small + more packages and languages)“ wählen: b ⏎
* Zurück ins Hauptmenu: R ⏎
* Schließlich, zum Installieren: I ⏎
* Root-Zugang beenden und den Installationsordner löschen:
    ```
    exit
    cd .. && rm -ir install-tl 
    ```

## Project
```
make install
```

# Build
## PDF
```
make pdf
```
