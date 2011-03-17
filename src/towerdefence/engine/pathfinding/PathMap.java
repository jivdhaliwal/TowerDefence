/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package towerdefence.engine.pathfinding;

import org.newdawn.slick.tiled.TiledMap;
import org.newdawn.slick.util.pathfinding.PathFindingContext;
import org.newdawn.slick.util.pathfinding.TileBasedMap;

/**
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class PathMap implements TileBasedMap {

    // The map created in tiled
    TiledMap map;

    // Grass = Terrain critters can walk on
    public static final int GRASS = 1;
    // Spire = Anything a plane has to navigate around
    public static final int SPIRE = 2;

    // Critter = ground walking enemy
    public static final int CRITTER = 3;

    // Plane = any flying enemy
    public static final int PLANE = 4;

    // Terrain settings for each tile
    private int[][] terrain;
    // Returns true if tile has been visited during search
    private boolean[][] visited;

    public PathMap(TiledMap map) {
        this.map = map;
        visited = new boolean[map.getWidth()][map.getHeight()];
        terrain = new int[map.getWidth()][map.getHeight()];

        // Set terrain values based on TiledMap
        setTerrain(map);
  
    }

    /*
     * Set terrain values by checking the map's collision layer
     */
    public void setTerrain(TiledMap map) {

        for (int x = 0; x < map.getWidth(); x++) {
            for (int y = 0; y < map.getHeight(); y++) {
                int tileID = map.getTileId(x, y, 0);

                String value = map.getTileProperty(tileID, "Type", "false");
                if ("grass".equals(value)) {
                    terrain[x][y] = GRASS;
                }
            }
        }
    }

    /*
     * Clear visited array
     */
    public void clearVisited() {
        for (int x = 0; x < getWidthInTiles(); x++) {
            for (int y = 0; y < getHeightInTiles(); y++) {
                visited[x][y] = false;
            }
        }
    }

    public boolean visited(int x, int y) {
        return visited[x][y];
    }

    public int getTerrain(int x, int y) {
        return terrain[x][y];
    }

    public boolean blocked(PathFindingContext context, int tx, int ty) {
        
        int unit = ((UnitMover) context.getMover()).getType();

        // planes can move anywhere
        if (unit == PLANE) {
            return false;
        }
        // tanks can only move across grass
        if (unit == CRITTER) {
            return terrain[tx][ty] != GRASS;
        }

        // unknown unit so everything blocks
        return true;

    }

    public int getWidthInTiles() {
        return map.getWidth();
    }

    public int getHeightInTiles() {
        return map.getHeight();
    }

    public void pathFinderVisited(int x, int y) {
        visited[x][y] = true;
    }

    public float getCost(PathFindingContext context, int tx, int ty) {
        return 1;
    }

}
