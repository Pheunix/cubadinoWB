import sys
import math
import random

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from enum import Enum

# ======================================
#            CONFIGURATION ENUM
# ======================================

move_queue = []  # File d'attente pour les mouvements à appliquer

class MoveConfig(Enum):
    """
    Paramètres de synchronisation de l'animation.
    Ces valeurs contrôlent la durée et la fréquence des mouvements du cube.
    """
    ANIMATION_DURATION_MS = 400  # Durée d'un mouvement en millisecondes
    FPS = 60                     # Nombre d'images par seconde pour l'animation
    MOVE_INTERVAL_MS = 400       # Intervalle entre les mouvements aléatoires (en ms)

class CubeConfig(Enum):
    """
    Paramètres visuels et géométriques du cube.
    Ces valeurs définissent l'apparence et la disposition du cube dans l'espace.
    """
    CUBE_SIZE = 3.0              # Taille d'un côté du cube
    VIEW_DISTANCE = -10          # Distance de la caméra par rapport au cube
    BACKGROUND_COLOR = (0.0, 0.0, 0.0, 1.0)  # Couleur de fond de la scène
    LIGHT_POSITION = (5, 5, 10, 1)           # Position de la source lumineuse
    LIGHT_AMBIENT = (0.2, 0.2, 0.2, 1)       # Lumière ambiante
    LIGHT_DIFFUSE = (0.8, 0.8, 0.8, 1)       # Lumière diffuse

class DisplayConfig(Enum):
    """
    Paramètres d'affichage du texte et des boutons.
    Ces valeurs définissent l'apparence des éléments de l'interface utilisateur.
    """
    TEXT_FONT_SIZE = 60          # Taille de la police pour le texte principal (augmentée)
    BUTTON_FONT_SIZE = 30        # Taille de la police pour les boutons (augmentée)
    BUTTON_WIDTH = 200           # Largeur des boutons
    BUTTON_HEIGHT = 60           # Hauteur des boutons
    BUTTON_BG_COLOR = (0.0, 0.0, 0.0)  # Couleur de fond des boutons
    BUTTON_BORDER_COLOR = (0.5, 1.0, 0.5)  # Couleur de la bordure des boutons

class FaceColor(Enum):
    """
    Couleurs RVB pour les faces du cube.
    Chaque face du cube est associée à une couleur spécifique.
    """
    WHITE = (1.0, 1.0, 1.0)
    YELLOW = (1.0, 1.0, 0.0)
    GREEN = (0.0, 1.0, 0.0)
    BLUE = (0.0, 0.0, 1.0)
    ORANGE = (1.0, 0.5, 0.0)
    RED = (1.0, 0.0, 0.0)
    BLACK = (0.0, 0.0, 0.0)

# --------------------------------------
# Définitions initiales des couleurs des faces du cube
# --------------------------------------

# Couleurs initiales pour le premier cube (cube1)
CUBE1_FACE_COLORS = {
    'U': [FaceColor.WHITE.value] * 9,  # Face du haut
    'D': [FaceColor.YELLOW.value] * 9, # Face du bas
    'F': [FaceColor.GREEN.value] * 9,  # Face avant
    'B': [FaceColor.BLUE.value] * 9,   # Face arrière
    'L': [FaceColor.ORANGE.value] * 9, # Face gauche
    'R': [FaceColor.RED.value] * 9,    # Face droite
}

# Couleurs initiales pour le second cube (cube2), toutes noires
CUBE2_FACE_COLORS = {face: [FaceColor.BLACK.value] * 9 for face in 'UDFBLR'}

# Couleurs cibles pour le second cube après coloration
TARGET_CUBE2_COLORS = {
    'U': [FaceColor.RED.value, FaceColor.ORANGE.value, FaceColor.WHITE.value,
            FaceColor.WHITE.value, FaceColor.GREEN.value, FaceColor.WHITE.value,
            FaceColor.WHITE.value, FaceColor.WHITE.value, FaceColor.WHITE.value],
    'D': [FaceColor.YELLOW.value, FaceColor.YELLOW.value, FaceColor.YELLOW.value,
            FaceColor.GREEN.value, FaceColor.YELLOW.value, FaceColor.YELLOW.value,
            FaceColor.YELLOW.value, FaceColor.YELLOW.value, FaceColor.YELLOW.value],
    'F': [FaceColor.BLUE.value, FaceColor.WHITE.value, FaceColor.GREEN.value,
            FaceColor.GREEN.value, FaceColor.ORANGE.value, FaceColor.YELLOW.value,
            FaceColor.GREEN.value, FaceColor.ORANGE.value, FaceColor.GREEN.value],
    'B': [FaceColor.BLUE.value, FaceColor.BLUE.value, FaceColor.GREEN.value,
            FaceColor.ORANGE.value, FaceColor.BLUE.value, FaceColor.BLUE.value,
            FaceColor.BLUE.value, FaceColor.BLUE.value, FaceColor.BLUE.value],
    'L': [FaceColor.ORANGE.value, FaceColor.ORANGE.value, FaceColor.WHITE.value,
            FaceColor.ORANGE.value, FaceColor.RED.value, FaceColor.ORANGE.value,
            FaceColor.ORANGE.value, FaceColor.ORANGE.value, FaceColor.ORANGE.value],
    'R': [FaceColor.WHITE.value, FaceColor.RED.value, FaceColor.BLUE.value,
            FaceColor.RED.value, FaceColor.GREEN.value, FaceColor.RED.value,
            FaceColor.RED.value, FaceColor.RED.value, FaceColor.RED.value]
}

# Séquence de mouvements prédéfinis pour le cube
POSSIBLE_MOOVE = [
    "R", "R'", "R2", "L", "L'", "L2",
    "U", "U'", "U2", "D", "D'", "D2",
    "F", "F'", "F2", "B", "B'", "B2",
]
PREDEFINED_SEQUENCE = [
    "R", "R'", "R2", "L", "L'", "L2",
    "U", "U'", "U2", "D", "D'", "D2",
    "F", "F'", "F2", "B", "B'", "B2","U","B'"
]

# ======================================
#          UTILITAIRES DE MATRICE
# ======================================

def identity_matrix():
    """Retourne une matrice identité 3×3, utilisée pour les transformations sans rotation."""
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

def rotation_matrix(axis, angle):
    """
    Retourne une matrice de rotation autour de l'axe donné ('x','y','z').
    L'angle est spécifié en degrés.
    """
    rad = math.radians(angle)  # Conversion de l'angle de degrés en radians

    if axis == 'x':
        return [
            [1, 0, 0],
            [0, math.cos(rad), -math.sin(rad)],
            [0, math.sin(rad),  math.cos(rad)],
        ]

    if axis == 'y':
        return [
            [ math.cos(rad), 0, math.sin(rad)],
            [0, 1, 0],
            [-math.sin(rad), 0, math.cos(rad)],
        ]

    if axis == 'z':
        return [
            [math.cos(rad), -math.sin(rad), 0],
            [math.sin(rad),  math.cos(rad), 0],
            [0, 0, 1],
        ]

    return identity_matrix()

def matrix_mult(A, B):
    """Multiplie deux matrices 3×3 pour obtenir une transformation combinée."""
    return [
        [sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]

def matrix_vector_mult(M, v):
    """Applique une matrice 3×3 à un vecteur pour obtenir une nouvelle position/orientation."""
    return (
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
    )

def rotate_point(p, axis, angle):
    """Fait tourner un point p=(x,y,z) autour d'un axe selon un angle donné."""
    th = math.radians(angle)  # Conversion de l'angle en radians
    x, y, z = p

    if axis == 'x':
        return (
            x,
            y * math.cos(th) - z * math.sin(th),
            y * math.sin(th) + z * math.cos(th),
        )

    if axis == 'y':
        return (
            x * math.cos(th) + z * math.sin(th),
            y,
            -x * math.sin(th) + z * math.cos(th),
        )

    if axis == 'z':
        return (
            x * math.cos(th) - y * math.sin(th),
            x * math.sin(th) + y * math.cos(th),
            z,
        )

    return (x, y, z)

# ======================================
#             CLASSE CUBIE
# ======================================

class Cubie:
    """Représente une seule petite pièce de cube, avec sa position, orientation et couleurs."""

    def __init__(self, x, y, z, size, face_colors):
        self.x, self.y, self.z = x * size, y * size, z * size  # Position initiale
        self.size = size  # Taille d'un cubie
        self.orientation = identity_matrix()  # Orientation initiale (aucune rotation)
        self.faces = {}  # Dictionnaire pour stocker les couleurs des faces visibles

        # Déterminer les couleurs des faces visibles en fonction de la position
        for face, cols in face_colors.items():
            belongs = (
                (face == 'U' and y == 1) or
                (face == 'D' and y == -1) or
                (face == 'F' and z == -1) or
                (face == 'B' and z == 1) or
                (face == 'L' and x == -1) or
                (face == 'R' and x == 1)
            )
            if not belongs:
                continue

            # Calculer l'index de la couleur dans la face
            if face in ('U', 'D'):
                col = int(x + 1)
                row = int((1 - z) if face == 'U' else (z + 1))

            elif face == 'F':
                col = int(x + 1)
                row = int(1 - y)

            elif face == 'B':
                col = int(1 - x)
                row = int(1 - y)

            elif face == 'L':
                col = int(1 - z)
                row = int(1 - y)

            else:  # 'R'
                col = int(z + 1)
                row = int(1 - y)

            idx = col + 3 * row
            self.faces[face] = cols[idx]

    def get_vertices(self):
        """Retourne les 8 sommets du cubie après application de l'orientation."""
        half = self.size / 2  # Demi-taille du cubie
        pts = [
            (-half, -half, -half), (half, -half, -half),
            (half,  half, -half), (-half,  half, -half),
            (-half, -half,  half), (half, -half,  half),
            (half,  half,  half), (-half,  half,  half),
        ]
        out = []
        for p in pts:
            vx, vy, vz = matrix_vector_mult(self.orientation, p)
            out.append((vx + self.x, vy + self.y, vz + self.z))
        return out

    def draw(self):
        """Dessine le cubie avec des faces colorées et des bords noirs."""
        verts = self.get_vertices()  # Obtenir les sommets orientés
        face_map = {
            'U': (3, 2, 6, 7), 'D': (0, 1, 5, 4),
            'F': (0, 1, 2, 3), 'B': (7, 6, 5, 4),
            'L': (0, 3, 7, 4), 'R': (1, 2, 6, 5),
        }

        # Remplir les faces avec des couleurs
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1, 1)
        glBegin(GL_QUADS)
        for face, idxs in face_map.items():
            glColor3fv(self.faces.get(face, (0, 0, 0)))  # Couleur de la face
            for i in idxs:
                glVertex3fv(verts[i])  # Dessiner chaque sommet de la face
        glEnd()
        glDisable(GL_POLYGON_OFFSET_FILL)

        # Dessiner les bords noirs pour délimiter les faces
        glColor3f(0, 0, 0)
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        glBegin(GL_LINES)
        for e0, e1 in edges:
            glVertex3fv(verts[e0])
            glVertex3fv(verts[e1])
        glEnd()

    def apply_rotation(self, axis, angle, center):
        """Fait tourner le cubie autour d'un centre donné selon un angle et un axe."""
        cx, cy, cz = center  # Centre de rotation
        tx, ty, tz = self.x - cx, self.y - cy, self.z - cz  # Translation vers l'origine
        rx, ry, rz = rotate_point((tx, ty, tz), axis, angle)  # Rotation

        # Mettre à jour la position et l'orientation
        self.x, self.y, self.z = rx + cx, ry + cy, rz + cz
        self.orientation = matrix_mult(rotation_matrix(axis, angle), self.orientation)

# ======================================
#         CLASSE RUBIKSCUBE
# ======================================

class RubiksCube:
    """Gère l'état du Rubik's cube, les mouvements, la coloration et le rendu."""

    def __init__(self, face_colors):
        # Couleurs initiales et dynamiques
        self.initial_colors = face_colors  # Couleurs initiales des faces
        self.current_colors = {f: cols[:] for f, cols in face_colors.items()}  # Couleurs actuelles
        self.init_cubies(self.current_colors)  # Initialiser les cubies
        self.current_move = None  # Mouvement actuel en cours

    """
        # État de l'animation de coloration
        self.color_faces = ['B', 'L', 'F', 'R', 'U', 'D']  # Ordre de coloration des faces
        self.next_color_index = 0  # Index de la prochaine face à colorier
        self.is_colorizing = False  # Indique si la coloration est en cours
        self.phase = None  # Phase actuelle de la coloration ('rotate' ou 'pause')
        self.phase_start = 0  # Temps de début de la phase actuelle
        self.rotate_duration = 500  # Durée de la rotation de la caméra (ms)
        self.pause_duration = 1000  # Durée de la pause entre les colorations (ms)
        self.camera_initial = None  # Position initiale de la caméra
    """

    def init_cubies(self, face_colors):
        """Instancie 27 objets Cubie en fonction des couleurs des faces."""
        size = CubeConfig.CUBE_SIZE.value / 3  # Taille d'un cubie
        self.cubies = [Cubie(x, y, z, size, face_colors)
                        for x in (-1, 0, 1)
                        for y in (-1, 0, 1)
                        for z in (-1, 0, 1)]

    def start_colorize(self, camera):
        """Commencer l'animation de coloration progressive des faces."""
        self.current_colors = {f: CUBE2_FACE_COLORS[f][:] for f in CUBE2_FACE_COLORS}
        self.init_cubies(self.current_colors)
        self.next_color_index = 0
        self.is_colorizing = True
        self.phase = 'rotate'
        self.phase_start = pygame.time.get_ticks()
        self.camera_initial = (camera.rot_x, camera.rot_y)

    def update_colorize(self, camera, Target_color=TARGET_CUBE2_COLORS):
        """Mettre à jour la coloration : faire tourner la caméra puis peindre chaque face."""
        if not self.is_colorizing:
            return

        # Angles de la caméra pour chaque face
        FACE_ANGLES = {'B': (20, 0), 'L': (20, 90), 'F': (20, 180),
                        'R': (20, 270), 'U': (60, 270), 'D': (-60, 270)}
        now = pygame.time.get_ticks()  # Temps actuel
        face = self.color_faces[self.next_color_index]  # Face actuelle à colorier

        # Phase de rotation de la caméra
        if self.phase == 'rotate':
            sx, sy = self.camera_initial  # Position initiale de la caméra
            tx, ty = FACE_ANGLES[face]  # Position cible de la caméra
            t = min((now - self.phase_start) / self.rotate_duration, 1)  # Progression de la rotation
            camera.rot_x = sx + (tx - sx) * t  # Interpolation de la rotation
            camera.rot_y = sy + (ty - sy) * t

            if t >= 1:
                self.phase = 'pause'
                self.phase_start = now

        # Phase de coloration
        else:
            self.current_colors[face] = Target_color[face][:]  # Mettre à jour les couleurs
            self.init_cubies(self.current_colors)  # Réinitialiser les cubies avec les nouvelles couleurs
            
            if now - self.phase_start >= self.pause_duration:
                self.next_color_index += 1
                if self.next_color_index >= len(self.color_faces):
                    self.is_colorizing = False
                    camera.rot_x, camera.rot_y = 20, -30  # Réinitialiser la caméra
                else:
                    self.phase = 'rotate'
                    self.phase_start = now
                    self.camera_initial = (camera.rot_x, camera.rot_y)

    def get_face_cubies(self, face):
        """Retourne les cubies appartenant à une face donnée pour la rotation."""
        eps = self.cubies[0].size / 2 + 1e-6  # Tolérance pour la sélection des cubies
        if face == 'F':
            return [c for c in self.cubies if abs(c.z + c.size) < eps]
        if face == 'B':
            return [c for c in self.cubies if abs(c.z - c.size) < eps]
        if face == 'U':
            return [c for c in self.cubies if abs(c.y - c.size) < eps]
        if face == 'D':
            return [c for c in self.cubies if abs(c.y + c.size) < eps]
        if face == 'L':
            return [c for c in self.cubies if abs(c.x + c.size) < eps]
        if face == 'R':
            return [c for c in self.cubies if abs(c.x - c.size) < eps]
        return []

    def start_move(self, move):
        """Initialise un mouvement de rotation de face pour l'animation."""
        if not move:
            return
        face, mod = move[0], move[1:]  # Face et modificateur du mouvement
        mult = {'': 1, "'": -1, '2': 2}.get(mod, 1)  # Multiplicateur pour l'angle

        # Axe et pivot pour chaque face
        axis, pivot_map = {
            'F': ('z', (0, 0, -self.cubies[0].size)),
            'B': ('z', (0, 0,  self.cubies[0].size)),
            'U': ('y', (0,  self.cubies[0].size, 0)),
            'D': ('y', (0, -self.cubies[0].size, 0)),
            'L': ('x', (-self.cubies[0].size, 0, 0)),
            'R': ('x', ( self.cubies[0].size, 0, 0)),
        }[face]
        angle = 90 * mult  # Angle total de rotation

        self.current_move = {
            'axis': axis,
            'pivot': pivot_map,
            'total': angle,
            'start_time': pygame.time.get_ticks(),
            'duration': MoveConfig.ANIMATION_DURATION_MS.value,
            'angle_so_far': 0,
            'affected': self.get_face_cubies(face)
        }

    def update_move(self):
        """Progresser l'animation du mouvement et finaliser la rotation."""
        m = self.current_move
        if not m:
            return

        elapsed = pygame.time.get_ticks() - m['start_time']  # Temps écoulé
        frac = min(elapsed / m['duration'], 1)  # Progression du mouvement
        m['angle_so_far'] = m['total'] * frac  # Angle actuel de rotation

        if frac >= 1:
            for c in m['affected']:
                c.apply_rotation(m['axis'], m['total'], m['pivot'])
            self.current_move = None

    def draw(self):
        """Dessine tous les cubies, appliquant les rotations de mouvement en cours."""
        for cubie in self.cubies:
            glPushMatrix()
            if self.current_move and cubie in self.current_move['affected']:
                axis = self.current_move['axis']
                angle = self.current_move['angle_so_far']
                axes = {'x': (1,0,0), 'y': (0,1,0), 'z': (0,0,1)}[axis]
                glRotatef(angle, *axes)
            cubie.draw()
            glPopMatrix()
    
    def apply_move(self, move):
        """Applique un mouvement spécifique au cube."""
        if not self.current_move:
            self.start_move(move)

# ======================================
#             CLASSE CAMERA
# ======================================

class Camera:
    """Gère l'orientation et la distance de la caméra dans la scène 3D."""

    def __init__(self):
        self.rot_x, self.rot_y = 20, -30  # Rotation initiale de la caméra
        self.distance = CubeConfig.VIEW_DISTANCE.value  # Distance initiale de la caméra

    def apply(self):
        """Applique les transformations de la caméra à OpenGL."""
        glTranslatef(0, 0, self.distance)  # Translation de la caméra
        glRotatef(self.rot_x, 1, 0, 0)  # Rotation autour de l'axe X
        glRotatef(self.rot_y, 0, 1, 0)  # Rotation autour de l'axe Y

# ======================================
#      RENDU DU TEXTE ET DES BOUTONS
# ======================================

def draw_text(pos, text, font_size):
    """Rendre du texte 2D en utilisant les polices Pygame dans un contexte OpenGL."""
    font = pygame.font.SysFont("Arial", font_size, bold=True)  # Police utilisée
    surf = font.render(text, True, (255, 200, 0), (0, 0, 0))  # Rendu du texte
    w, h = surf.get_width(), surf.get_height()  # Dimensions du texte
    data = pygame.image.tostring(surf, "RGBA", True)  # Conversion en chaîne d'octets

    # Passer en projection orthographique pour dessiner le texte
    glMatrixMode(GL_PROJECTION)
    glPushMatrix(); glLoadIdentity()
    glOrtho(0, pygame.display.get_surface().get_width(),
            pygame.display.get_surface().get_height(), 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix(); glLoadIdentity()

    # Dessiner les pixels du texte
    glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)
    glRasterPos2i(pos[0], pos[1])
    glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)

    # Restaurer la perspective
    glPopMatrix()
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_button(pos, label):
    """Dessine un bouton cliquable avec le texte 'label'."""
    # Récupérer la position du coin supérieur gauche
    x, y = pos
    # Largeur et hauteur du bouton
    w = DisplayConfig.BUTTON_WIDTH.value
    h = DisplayConfig.BUTTON_HEIGHT.value

    # --- Passage en projection orthographique ---
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(
        0,
        pygame.display.get_surface().get_width(),
        pygame.display.get_surface().get_height(),
        0,
        -1,
        1
    )
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # Désactiver éclairage et profondeur pour l'UI
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)

    # --- Fond du bouton ---
    glColor3fv(DisplayConfig.BUTTON_BG_COLOR.value)
    glBegin(GL_QUADS)
    glVertex2f(x,   y)
    glVertex2f(x+w, y)
    glVertex2f(x+w, y+h)
    glVertex2f(x,   y+h)
    glEnd()

    # --- Bordure du bouton ---
    glColor3fv(DisplayConfig.BUTTON_BORDER_COLOR.value)
    glLineWidth(2)
    glBegin(GL_LINE_LOOP)
    glVertex2f(x,   y)
    glVertex2f(x+w, y)
    glVertex2f(x+w, y+h)
    glVertex2f(x,   y+h)
    glEnd()

    # --- Texte centré dans le bouton ---
    font = pygame.font.SysFont(
        "Arial",
        DisplayConfig.BUTTON_FONT_SIZE.value
    )
    tw, th = font.size(label)
    tx = x + (w - tw) // 2
    ty = y + h - th // 3

    # Utiliser la même couleur pour le texte que pour le fond du bouton
    glColor3fv(DisplayConfig.BUTTON_BG_COLOR.value)
    draw_text((tx-25, ty+1), label, DisplayConfig.BUTTON_FONT_SIZE.value + 5)

    # Réactiver éclairage et profondeur
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)

    # Restaurer matrices
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_progress_bar(pos, width, height, progress):
    """Dessine une barre de progression."""
    x, y = pos

    # --- Passage en projection orthographique ---
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(
        0,
        pygame.display.get_surface().get_width(),
        pygame.display.get_surface().get_height(),
        0,
        -1,
        1
    )
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # Désactiver éclairage et profondeur pour l'UI
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)

    # --- Fond de la barre de progression ---
    glColor3f(0.3, 0.3, 0.3)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + width, y)
    glVertex2f(x + width, y + height)
    glVertex2f(x, y + height)
    glEnd()

    # --- Barre de progression ---
    glColor3f(0.0, 1.0, 0.0)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + width * progress, y)
    glVertex2f(x + width * progress, y + height)
    glVertex2f(x, y + height)
    glEnd()

    # Réactiver éclairage et profondeur
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)

    # Restaurer matrices
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

# ======================================
#         INITIALISATION
# ======================================

def init_window():
    """Initialiser Pygame, le contexte OpenGL et retourner la taille de la fenêtre."""
    pygame.init()  # Initialisation de Pygame
    info = pygame.display.Info()  # Informations sur l'affichage
    width, height = info.current_w, info.current_h  # Taille de la fenêtre
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | FULLSCREEN)  # Mode plein écran
    pygame.display.set_caption("Rubik's Cube — Plein Écran")  # Titre de la fenêtre

    glClearColor(*CubeConfig.BACKGROUND_COLOR.value)  # Couleur de fond
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, width/height, 0.1, 100)  # Perspective
    glMatrixMode(GL_MODELVIEW)

    glEnable(GL_DEPTH_TEST)  # Activer le test de profondeur
    glEnable(GL_LIGHTING); glEnable(GL_LIGHT0)  # Activer l'éclairage
    glLightfv(GL_LIGHT0, GL_POSITION, CubeConfig.LIGHT_POSITION.value)
    glLightfv(GL_LIGHT0, GL_AMBIENT,  CubeConfig.LIGHT_AMBIENT.value)
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  CubeConfig.LIGHT_DIFFUSE.value)
    glEnable(GL_COLOR_MATERIAL)  # Activer le matériau coloré
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    return width, height

# ======================================
#       GESTION DES ÉVÉNEMENTS ET MISE À JOUR DE L'ÉTAT
# ======================================

def handle_mouse_click(pos, current_page, states, cube2, cam, btn_toggle, btn_color, btn_start):
    """Gérer les clics de souris et mettre à jour les pages d'état de l'interface utilisateur et les actions."""
    mx, my = pos  # Position du clic
    w_btn, h_btn = DisplayConfig.BUTTON_WIDTH.value, DisplayConfig.BUTTON_HEIGHT.value  # Taille des boutons

    # Bouton de basculement de page
    tx, ty = btn_toggle
    if tx <= mx <= tx + w_btn and ty <= my <= ty + h_btn:
        # Basculer entre la page 1 et 2
        current_page = 2 if current_page == 1 else 1
        states.update({
            'started_moves': False,
            'colorized': False,
            'seq_idx': 0,
            'seq_complete': False
        })
        cube2.init_cubies(CUBE2_FACE_COLORS)  # Réinitialiser les cubies
        cam.__init__()  # Réinitialiser la caméra

        # Réinitialiser le minuteur du cube1 lors du retour à la page 1
        pygame.time.set_timer(USEREVENT, MoveConfig.MOVE_INTERVAL_MS.value)
        return current_page

    # Boutons de la page 2
    if current_page == 2:
        # Bouton de coloration
        cx, cy = btn_color
        if cx <= mx <= cx + w_btn and cy <= my <= cy + h_btn:
            if not cube2.is_colorizing and not states['colorized']:
                cube2.start_colorize(cam)
                states['colorized'] = True
            return current_page

        # Bouton de démarrage de la séquence
        sx, sy = btn_start
        if sx <= mx <= sx + w_btn and sy <= my <= sy + h_btn:
            if states['colorized'] and not cube2.is_colorizing and not states['started_moves']:
                states['started_moves'] = True
                states['seq_idx'] = 0  # Réinitialiser l'index de la séquence
                states['seq_complete'] = False
            return current_page

    return current_page

def process_events(cube1, cube2, cam, states, btn_toggle, btn_color, btn_start):
    """Interroger les événements Pygame et déclencher des mouvements ou des mises à jour de l'interface utilisateur."""
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit(); sys.exit()  # Quitter l'application

        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            states['current_page'] = handle_mouse_click(
                event.pos, states['current_page'], states,
                cube2, cam, btn_toggle, btn_color, btn_start
            )
    """
        # Déclencher des mouvements aléatoires pour cube1
        if event.type == USEREVENT and not cube1.current_move:
            cube1.start_move(random.choice(POSSIBLE_MOOVE))
    """

def update_states(cube1, cube2):
    """Mettre à jour les animations pour les deux cubes en fonction de la page actuelle."""
    cube1.update_move()  # Mettre à jour le mouvement du cube1
    cube2.update_move()  # Mettre à jour le mouvement du cube2

    # Appliquer les mouvements explicites depuis la file d'attente
    if not cube1.current_move and move_queue:
        next_move = move_queue.pop(0)  # Récupérer le prochain mouvement
        cube1.start_move(next_move)

# ======================================
#          RENDU DES PAGES
# ======================================

def render_page1(cube1):
    """Rendre la page 1 : démonstration du solveur de Rubik's Cube en rotation."""
    glPushMatrix()
    glTranslatef(-4.5, -1, 0)  # Translation du cube
    draw_text((600, 100), "Solveur de Rubik's Cube", DisplayConfig.TEXT_FONT_SIZE.value)
    draw_text((900, 300), "Projet Solveur de Rubik’s Cube – Promo 2024-2025 (Version Bêta)", DisplayConfig.TEXT_FONT_SIZE.value // 2)
    draw_text((900, 350), "Equipe :", DisplayConfig.TEXT_FONT_SIZE.value // 2)
    draw_text((1000, 400), "Maxime Chantrainne (chef de projet)", DisplayConfig.TEXT_FONT_SIZE.value // 2)
    draw_text((1000, 450), "Massimo Marcelin", DisplayConfig.TEXT_FONT_SIZE.value // 2)
    draw_text((1000, 500), "Enzo Macajone", DisplayConfig.TEXT_FONT_SIZE.value // 2)
    draw_text((1000, 550), "Wandrille Berne", DisplayConfig.TEXT_FONT_SIZE.value // 2)
    draw_text((1000, 600), "Maxence Dufour", DisplayConfig.TEXT_FONT_SIZE.value // 2)
    draw_text((1000, 650), "Lucas Farran", DisplayConfig.TEXT_FONT_SIZE.value // 2)
    draw_text((900, 700), "Machine basé sur le projet open source nommé Cubotino", DisplayConfig.TEXT_FONT_SIZE.value // 2)
    cube1.draw()  # Dessiner le cube
    glPopMatrix()

def render_page2(cube2, cam, states, btn_color, btn_start, btn_toggle):
    """Rendre la page 2 : démonstration interactive de coloration et de séquence."""
    glPushMatrix()
    glScalef(1.5, 1.5, 1.5)  # Mise à l'échelle du cube
    cube2.draw()  # Dessiner le cube
    glPopMatrix()

    # Superposition de l'interface utilisateur
    glDisable(GL_LIGHTING)
    draw_button(btn_color, "Coloriser")  # Bouton de coloration
    draw_button(btn_start, "Démarrer")  # Bouton de démarrage

    # Messages d'état
    if not states['colorized']:
        draw_text((30, 100), "Appuyer sur 'Coloriser'", DisplayConfig.TEXT_FONT_SIZE.value)
    elif cube2.is_colorizing:
        draw_text((30, 60), f"Colorisation: Face {cube2.color_faces[cube2.next_color_index]}",
                    DisplayConfig.TEXT_FONT_SIZE.value)
    elif states['seq_complete']:
        draw_text((30, 100), "Séquence terminée !", DisplayConfig.TEXT_FONT_SIZE.value)
    elif states['started_moves']:
        draw_text((30, 90), "Exécution de la séquence…", DisplayConfig.TEXT_FONT_SIZE.value)
        # Dessiner la barre de progression
        progress = states['seq_idx'] / len(PREDEFINED_SEQUENCE)
        draw_progress_bar((900, 40), 700, 40, progress)
    else:
        draw_text((30, 100), "Colorisation terminée ! Appuyer sur 'Démarrer'",
                    DisplayConfig.TEXT_FONT_SIZE.value)

    glEnable(GL_LIGHTING)

def apply_cube_move(move):
    """
    Ajoute un mouvement spécifique à la file d'attente.
    
    Arguments:
    - move: Mouvement à appliquer (par exemple, "R", "U'", "F2").
    """
    move_queue.append(move)

# ======================================
#            BOUCLE PRINCIPALE
# ======================================

def render_loop(cube1, cube2, cam, states, btn_toggle, btn_color, btn_start):
    """Boucle de rendu et de mise à jour continue."""
    clock = pygame.time.Clock()
    while True:
        clock.tick(MoveConfig.FPS.value)
        process_events(cube1, cube2, cam, states, btn_toggle, btn_color, btn_start)
        update_states(cube1, cube2)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        cam.apply()

        if states['current_page'] == 1:
            render_page1(cube1)
        else:
            render_page2(cube2, cam, states, btn_color, btn_start, btn_toggle)

        draw_button(btn_toggle, f"Résoudre" if states['current_page'] == 1 else "Menu")
        pygame.display.flip()

def main(cube1_colors=None, cube2_colors=None):
    """
    Initialise la fenêtre, les objets et démarre la boucle de rendu.
    
    Arguments:
    - cube1_colors: Dictionnaire des couleurs des faces pour le premier cube.
    - cube2_colors: Dictionnaire des couleurs des faces pour le second cube.
    """
    width, height = init_window()
    cube1_colors = cube1_colors or CUBE1_FACE_COLORS
    cube2_colors = cube2_colors or CUBE2_FACE_COLORS

    cube1 = RubiksCube(cube1_colors)
    cube2 = RubiksCube(cube2_colors)
    cam = Camera()

    btn_toggle = (width - DisplayConfig.BUTTON_WIDTH.value - 20, 20)
    btn_color = (width - DisplayConfig.BUTTON_WIDTH.value - 20, height // 2 - DisplayConfig.BUTTON_HEIGHT.value)
    btn_start = (width - DisplayConfig.BUTTON_WIDTH.value - 20, height // 2 + DisplayConfig.BUTTON_HEIGHT.value)

    states = {
        'current_page': 1,
        'started_moves': False,
        'colorized': False,
        'seq_idx': 0,
        'seq_complete': False
    }

    pygame.time.set_timer(USEREVENT, MoveConfig.MOVE_INTERVAL_MS.value)
    render_loop(cube1, cube2, cam, states, btn_toggle, btn_color, btn_start)

if __name__ == "__main__":
    # Exemple d'initialisation avec des couleurs personnalisées
    custom_cube1_colors = {
        'U': [FaceColor.WHITE.value] * 9,
        'D': [FaceColor.YELLOW.value] * 9,
        'F': [FaceColor.GREEN.value] * 9,
        'B': [FaceColor.BLUE.value] * 9,
        'L': [FaceColor.ORANGE.value] * 9,
        'R': [FaceColor.RED.value] * 9,
    }

    custom_cube2_colors = {
        'U': [FaceColor.RED.value] * 9,
        'D': [FaceColor.ORANGE.value] * 9,
        'F': [FaceColor.BLUE.value] * 9,
        'B': [FaceColor.GREEN.value] * 9,
        'L': [FaceColor.YELLOW.value] * 9,
        'R': [FaceColor.WHITE.value] * 9,
    }

    # Lancer l'affichage avec des couleurs personnalisées
    main()
    apply_cube_move("R")
    apply_cube_move("U'")
    apply_cube_move("F2")
