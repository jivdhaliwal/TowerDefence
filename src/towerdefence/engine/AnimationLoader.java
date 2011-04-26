package towerdefence.engine;

import org.newdawn.slick.Animation;
import org.newdawn.slick.Image;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.SpriteSheet;

/**
 * Parses sprite sheets and creates animations
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class AnimationLoader {

    // Critter and tower types
    public final static int NORMAL = 0;
    public final static int FIRE = 1;
    public final static int ICE = 2;
    public final static int BOSS = 3;
    // Direction constants
    public static final int UP = 0;
    public static final int DOWN = 1;
    public static final int LEFT = 2;
    public static final int RIGHT = 3;
    
    private Animation up, down, left, right;

    public AnimationLoader() {
    }

    
    public Animation[] getCritterAnimation(int critterType) throws SlickException {
        Image normalSheet = null;
        if (critterType == NORMAL) {
            normalSheet = new Image("data/sprites/critters/antNormal.png");
        } else if (critterType == FIRE) {
            normalSheet = new Image("data/sprites/critters/antFire.png");
        } else if (critterType == ICE) {
            normalSheet = new Image("data/sprites/critters/antIce.png");
        }
        
        SpriteSheet critterSheet = new SpriteSheet(normalSheet, 64, 64);

        left = new Animation(critterSheet,0,0,7,0,true, 100, true);
        up = new Animation(critterSheet,0,2,7,2,true, 100, true);
        right = new Animation(critterSheet,0,4,7,4,true, 100, true);
        down = new Animation(critterSheet,0,6,7,6,true, 100, true);

        Animation[] critterAnimation = {up,down,left,right};

        return critterAnimation;
    }
    
    public Image[] getTowerSprites(int towerType) throws SlickException {
        Image[] tower = new Image[3];
        if(towerType==NORMAL) {
            tower[0] = new Image("data/sprites/towers/greentower.png");
            tower[1] = new Image("data/sprites/towers/arrow.png");
            tower[2] = new Image("data/sprites/laser/green.png");
        } else if(towerType==FIRE) {
            tower[0] = new Image("data/sprites/towers/redtower.png");
            tower[1] = new Image("data/sprites/towers/arrow.png");
            tower[2] = new Image("data/sprites/laser/red.png");
        } else if(towerType==ICE) {
            tower[0] = new Image("data/sprites/towers/bluetower.png");
            tower[1] = new Image("data/sprites/towers/arrow.png");
            tower[2] = new Image("data/sprites/laser/blue.png");  
        }
        
        return tower;
    }
}

