/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package towerdefence.states;

import org.newdawn.slick.Animation;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.Input;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.state.BasicGameState;
import org.newdawn.slick.state.StateBasedGame;
import org.newdawn.slick.tiled.TiledMap;

/**
 *
 * @author Jiv
 */
public class GameplayState extends BasicGameState {

    int stateID = -1;

    private boolean[][] blocked;
    private static final int TILESIZE = 32;

    private TiledMap map;
    private Animation sprite, up, down, left, right;
    private Image zombieSprites;
    private Image testerSprite;

    private float x,y;

    GameplayState(int stateID) {
        this.stateID = stateID;
    }

    @Override
    public int getID() {
        return 1;
    }

    public void init(GameContainer container, StateBasedGame game) throws SlickException {

        zombieSprites = new Image("data/sprites/zombie.png");
        testerSprite = new Image("data/sprites/tester.png");
        map = new TiledMap("data/maps/map1.tmx");
        Image [] movementUp = { zombieSprites.getSubImage(0, 2*128, 128, 128) };
        Image [] movementDown = { zombieSprites.getSubImage(0, 6*128, 128, 128) };
        Image [] movementLeft = { zombieSprites.getSubImage(0, 0*128, 128, 128) };
        Image [] movementRight = { zombieSprites.getSubImage(0, 4*128, 128, 128) };

        up = new Animation(movementUp, 1000);
        down = new Animation(movementDown, 1000);
        left = new Animation(movementLeft, 1000);
        right = new Animation(movementRight, 1000);

        // Initialise sprite direction and position
        sprite = left;
        x = TILESIZE*10;
        y = TILESIZE*10;
        
        // build a collision map based on tile properties in the TileD map
        blocked = new boolean[map.getWidth()][map.getHeight()];
        for (int xAxis = 0; xAxis < map.getWidth(); xAxis++) {
            for (int yAxis = 0; yAxis < map.getHeight(); yAxis++) {
                int tileID = map.getTileId(xAxis, yAxis, 0);
                
                String value = map.getTileProperty(tileID, "blocked", "false");
                if ("true".equals(value)) {
                    blocked[xAxis][yAxis] = true;
                }
            }
        }
        
    }

    public void render(GameContainer container, StateBasedGame game, Graphics g) throws SlickException {
        map.render(0,0,1,1,18,18);
        sprite.draw(x, y);
        testerSprite.draw(x,y);
    }

    public void update(GameContainer container, StateBasedGame game, int delta) throws SlickException {

        Input input = container.getInput();
        
        if (input.isKeyDown(Input.KEY_UP))
        {
            sprite = up;
            if(!isBlocked(x,y - 0.1f*delta))
            {
                y -= 0.1f*delta;
            }
        }
        else if (input.isKeyDown(Input.KEY_DOWN))
        {
            sprite = down;
            if(!isBlocked(x,y + 0.1f*delta))
            {
                y += 0.1f*delta;
            }
        }
        else if (input.isKeyDown(Input.KEY_LEFT))
        {
            sprite = left;
            if(!isBlocked(x - 0.1f*delta,y))
            {
                x -= 0.1f*delta;
            }
        }
        else if (input.isKeyDown(Input.KEY_RIGHT))
        {
            sprite = right;
            if(!isBlocked(x + 0.1f*delta,y))
            {
                x += 0.1f*delta;
            }
        }
         
         

    }

    private boolean isBlocked(float x, float y) {
        int xTile = (int) Math.floor((x / TILESIZE));
        int yTile = (int) Math.floor((y / TILESIZE));
        
        return blocked[xTile+1][yTile+1];
    }

}
