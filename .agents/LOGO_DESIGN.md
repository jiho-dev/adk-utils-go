# ADK-Utils Logo Design Notes

## Project Context

We are creating a logo for **adk-utils** based on the original **ADK (Agent Development Kit)** logo.

## Original ADK Logo (agent-development-kit.png)

The original ADK logo consists of:

### Top Part - Robot/Clip Head

- **Horizontal oval shape** like a robot head or paper clip
- **Left side**: blue stroke `#4285F4`
- **Right side**: green stroke `#34A853`
- **Two circular eyes** in blue inside the shape

### Bottom Part - Code Symbols

- **Bracket `[`** in yellow `#FBBC04`
- **Chevrons `<>`** in red `#EA4335`

## Color Palette

| Color  | Hex       | Usage                   |
| ------ | --------- | ----------------------- |
| Blue   | `#4285F4` | Left side head, eyes    |
| Green  | `#34A853` | Right side head         |
| Yellow | `#FBBC04` | Bracket `[`             |
| Red    | `#EA4335` | Chevrons `<>`           |
| Violet | `#9B59B6` | Floating shapes (added) |

---

## Current Design: Robotito

### Naming Convention

- **Robotito**: complete set (head + cuerpito/body)
- **Head**: blue/green oval shape with eyes
- **Cuerpito**: the 3 code symbols (`[`, `<`, `>`)

### Robotito Structure

#### Head

- Oval shape with blue left side and green right side
- Two circular blue eyes
- Stroke-width: 32
- Centered on canvas

#### Cuerpito (Body)

- **Bracket `[`** yellow (left)
- **Chevron `<`** red (center-left)
- **Chevron `>`** red (right)
- Horizontally distributed
- Separated from head (gap of ~75px in original coordinates)
- Stroke-width: 32 (same as head)
- Height: 120px (y=310 to y=430 in original coords)

### Transformation

Robotito is scaled to 60% and centered:

```
transform="translate(256, 256) scale(0.6) translate(-275, -227)"
```

---

## Floating Shapes (Utils)

Represent the project's **utilities/tools**. Distributed around robotito, without touching it (it's allergic to being touched).

### Current Shapes (V4)

| Position          | Shape    | Color  | Type    | Size   |
| ----------------- | -------- | ------ | ------- | ------ |
| Top center        | Circle   | Red    | Filled  | Tiny   |
| Top left          | Triangle | Violet | Filled  | Medium |
| Top right         | Square   | Yellow | Filled  | Tiny   |
| Top right lower   | Hexagon  | Green  | Filled  | Medium |
| Left upper        | Circle   | Blue   | Outline | Tiny   |
| Left              | Plus     | Yellow | Outline | Large  |
| Right             | Circle   | Red    | Filled  | Small  |
| Right lower       | Square   | Violet | Outline | Large  |
| Bottom left       | Circle   | Blue   | Filled  | Medium |
| Bottom left inner | Triangle | Green  | Filled  | Tiny   |
| Bottom right      | Circle   | Green  | Outline | Large  |
| Bottom center     | Square   | Blue   | Outline | Small  |

### Design Principles

- **Shape variety**: circle, square, triangle, hexagon, plus
- **Size variety**: tiny, small, medium, large
- **Style mix**: filled and outlined
- **Distribution**: all directions (top, bottom, sides, corners)
- **Spacing**: not too close to robotito

### Design Notes

- **Diamond and star** were discarded as they didn't fit visually
- **V4** has a more chaotic/organic distribution with varied sizes
- Top-left triangle was reduced for better visual balance

---

## Files

| File                                    | Description                                   |
| --------------------------------------- | --------------------------------------------- |
| `docs/images/logo.svg`                  | Main logo (512x512)                           |
| `docs/images/header.svg`                | Header for README (800x200, white background) |
| `docs/images/agent-development-kit.png` | Original ADK logo reference                   |

---

## SVG Technical Parameters

- **ViewBox**: `0 0 512 512` (main logo), `0 0 800 200` (header)
- **Stroke-width robotito**: 32
- **Stroke-width floating shapes**: 3-8
- **Stroke-linecap**: round
- **Stroke-linejoin**: round
- **Eye radius**: 18

---

## Change History

1. Removed original geometric shapes surrounding the logo
2. Cuerpito symbols distributed horizontally (more space)
3. Cuerpito symbols increased in height (120px)
4. Cuerpito stroke-width increased to 32 (same as head)
5. Right chevron moved further right
6. Cuerpito centered horizontally
7. Head and eyes centered to align with cuerpito
8. Robotito scaled to 60% to make room for floating shapes
9. Added varied floating shapes around
10. Added violet color to palette
11. Cuerpito moved down to separate from head
12. Floating shapes with variety: hexagon, plus, triangle
13. Mix of filled and outlined shapes
14. Created header version with text and white background

---

## Future Ideas

- [ ] Monochrome version
- [ ] Favicon version (simplified)
- [ ] SVG animation
